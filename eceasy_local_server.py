"""
This module implements a local-only backend server for ECEasy.
It replaces the Lepton AI dependency with local Ollama, OpenAI, or DeepSeek as the LLM provider.
"""

import json
import os

# Disable ChromaDB Telemetry to avoid errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Suppress logging from libraries that might be noisy
import logging
logging.getLogger("chromadb").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

import warnings
# Suppress Pydantic V2 deprecation warnings coming from ChromaDB
warnings.filterwarnings("ignore", message=".*Accessing the 'model_fields' attribute on the instance is deprecated.*")

import re
import shelve
import uuid
from contextlib import asynccontextmanager
from typing import List, Generator, Optional
from pydantic import BaseModel

# ======== FastAPI Imports ========
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import httpx
from loguru import logger

# ======== OpenAI / Ollama Imports ========
import openai

# ======== Search Engine Functions ========
try:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
except ImportError:
    logger.warning("duckduckgo_search / ddgs not installed. Web search will be disabled.")
    DDGS = None

# ======== Local Imports ========
import ecEasyPrompts

# ======== Image Support ========
try:
    from image_retrieval import ImageRetriever, suggest_images_for_response
    image_retriever = None  # Will be initialized at startup
except ImportError:
    logger.warning("image_retrieval module not found. Image suggestions will be disabled.")
    ImageRetriever = None
    suggest_images_for_response = None
    image_retriever = None

# ======== Load .env first — all os.environ.get() calls below will pick it up ========
from dotenv import load_dotenv
load_dotenv(override=True)

# ======== Configuration ========

# --- Server Config ---
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))

# --- Knowledge Base Selection ---
# Set KNOWLEDGE in .env:
#   "faiss"  → FAISS ECE knowledge base (./faiss_index_all-MiniLM-L6-v2/)
#   "chroma" → ChromaDB network knowledge base (./arag/chromaVectorStore/)
KNOWLEDGE = os.environ.get("KNOWLEDGE", "faiss").lower()

# --- UI Version ---
# Set UI_VERSION in .env:
#   "newUI" → new React/Vite chat interface (./newUI/)
#   "oldUI" → original Next.js search interface (./ui/)
#   "frontpage" → static demo landing page (./newDesign/FrontPage/)
UI_VERSION = os.environ.get("UI_VERSION", "newUI").lower()

# --- LLM Provider ---
# Set LLM_PROVIDER in .env: "ollama", "openai", or "deepseek"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()

# --- Feature Flags ---
KV_NAME = os.environ.get("KV_NAME", "eceasy-chat-local.kv")
REFERENCE_COUNT = 8
SHOULD_DO_RELATED_QUESTIONS = os.environ.get("RELATED_QUESTIONS", "true").lower() == "true"

# --- Provider Specific Config ---

# 1. Ollama Configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")

# 2. OpenAI Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", os.environ.get("LLM_REMOTE_OPENAI_API_KEY", ""))
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", os.environ.get("LLM_REMOTE_OPENAI_MODEL", "gpt-4o"))
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", os.environ.get("LLM_REMOTE_OPENAI_URL", "https://api.openai.com/v1"))

# 3. DeepSeek Configuration
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", os.environ.get("LLM_REMOTE_API_KEY", ""))
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", os.environ.get("LLM_REMOTE_URL", "https://api.deepseek.com"))
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", os.environ.get("LLM_REMOTE_MODEL", "deepseek-chat"))

def get_current_model_name():
    if LLM_PROVIDER == "openai":
        return OPENAI_MODEL
    elif LLM_PROVIDER == "deepseek":
        return DEEPSEEK_MODEL
    return OLLAMA_MODEL

LLM_MODEL = get_current_model_name()

logger.info(f"Knowledge base : {KNOWLEDGE.upper()}")
logger.info(f"UI Version     : {UI_VERSION}")
logger.info(f"LLM Provider   : {LLM_PROVIDER}, Model: {LLM_MODEL}")
logger.info(f"Server         : {HOST}:{PORT}")

# ======== Knowledge Base Import ========

if KNOWLEDGE == "faiss":
    try:
        from faiss_rag import get_rag_context
        logger.info(f"Knowledge base: FAISS (ECE knowledge — faiss)")
    except ImportError as e:
        logger.warning(f"Could not import faiss_rag: {e}. RAG functionality will be disabled.")
        def get_rag_context(query): return []
else:  # "chroma"
    try:
        from arag.arag import get_rag_context
        logger.info("Knowledge base: ChromaDB (Network knowledge — arag/chromaVectorStore/)")
    except ImportError as e:
        logger.warning(f"Could not import arag.arag: {e}. RAG functionality will be disabled.")
        def get_rag_context(query): return []

# Stop words for the LLM
# OpenAI API limits to 4 stop sequences.
STOP_WORDS = [
    "<|im_end|>",
    "[End]",
    "\nReferences:\n",
    "\nSources:\n",
]

# ======== Models ========

class QueryRequest(BaseModel):
    query: str
    search_uuid: str
    generate_related_questions: Optional[bool] = True
    llm_provider: Optional[str] = None
    api_key: Optional[str] = None
    use_server_key: Optional[bool] = None

class ImageSuggestion(BaseModel):
    path: str
    description: str
    doc_type: str
    source_relpath: str

class ChatResponse(BaseModel):
    text: Optional[str] = None
    contexts: Optional[List[dict]] = None
    related_questions: Optional[List[str]] = None
    suggested_images: Optional[List[ImageSuggestion]] = None
    flowchart: Optional[str] = None

# ======== Helper Functions ========

def get_model_name_for_provider(provider: str) -> str:
    if provider == "openai":
        return OPENAI_MODEL
    if provider == "deepseek":
        return DEEPSEEK_MODEL
    return OLLAMA_MODEL


def resolve_runtime_llm_config(request: QueryRequest) -> tuple[openai.OpenAI, str, str, bool]:
    """
    Resolve provider/model/client for this request.

    Returns: (client, provider, model, using_server_key)
    """
    provider = (request.llm_provider or LLM_PROVIDER).lower()
    if provider not in {"ollama", "openai", "deepseek"}:
        raise HTTPException(status_code=400, detail="llm_provider must be one of: ollama, openai, deepseek")

    if provider == "ollama":
        client = openai.OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",  # API key is not required for local Ollama
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=10.0),
        )
        return client, provider, OLLAMA_MODEL, True

    user_api_key = (request.api_key or "").strip()
    legacy_client_mode = (
        request.llm_provider is None
        and request.api_key is None
        and request.use_server_key is None
    )
    using_server_key = bool(request.use_server_key)

    # Keep old UI compatibility: if no new runtime fields are provided, use server-side key.
    if legacy_client_mode:
        using_server_key = True

    if not user_api_key and not using_server_key:
        raise HTTPException(
            status_code=400,
            detail="Provide api_key or set use_server_key=true for remote providers.",
        )

    if provider == "openai":
        api_key = OPENAI_API_KEY if using_server_key else user_api_key
        if not api_key:
            raise HTTPException(status_code=400, detail="OpenAI API key is required for the selected mode.")
        client = openai.OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
    else:
        api_key = DEEPSEEK_API_KEY if using_server_key else user_api_key
        if not api_key:
            raise HTTPException(status_code=400, detail="DeepSeek API key is required for the selected mode.")
        client = openai.OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    return client, provider, get_model_name_for_provider(provider), using_server_key

def search_with_duckduckgo(query: str) -> List[dict]:
    """
    Search using DuckDuckGo (via ddgs directly) and return formatted contexts.
    """
    if not DDGS:
        return []

    try:
        results = []
        # Add retry logic
        for attempt in range(3):
            try:
                with DDGS() as ddgs:
                    # max_results corresponds to 'max_results' in ddgs.text()
                    # use a slightly larger number to ensure we get enough valid ones
                    ddgs_gen = ddgs.text(query, max_results=REFERENCE_COUNT)
                    if ddgs_gen:
                        results = list(ddgs_gen)
                        if results:
                            break # Success
            except Exception as e:
                logger.warning(f"DuckDuckGo attempt {attempt+1} failed: {e}")
                # Optional: time.sleep(1)

        logger.info(f"DuckDuckGo found {len(results)} results")

        if results:
             return [
                {
                    "id": str(uuid.uuid4()),
                    "name": r.get("title", "Source"),
                    "url": r.get("href", "#"), # ddgs uses 'href' usually
                    "snippet": r.get("body", "") # ddgs uses 'body' usually
                }
                for r in results
            ]

        return []
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return []

def get_related_questions(
    query: str,
    contexts: List[dict],
    client: openai.OpenAI,
    model_name: str,
) -> List[str]:
    """
    Generates related questions using the local LLM.
    """
    if not contexts:
        return []

    context_text = "\n\n".join([c["snippet"] for c in contexts])[:4000] # Limit context size

    prompt = ecEasyPrompts._more_questions_prompt.format(context=context_text)
    prompt += f"\n{query}"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=256,
            temperature=0.7,
        )
        if not response.choices:
            return []
        content = response.choices[0].message.content

        # Log raw content for debugging
        logger.info(f"Related questions raw output: {content}")

        # Parse the output. We expect a list of questions, but LLM might be chatty.
        # We try to extract lines that look like questions or JSON.
        questions = []
        for line in content.split('\n'):
            line = line.strip()
            if line and (line.endswith('?') or line.startswith('-') or line.startswith('*') or line[0].isdigit()):
                # Clean up bullets
                line = re.sub(r"^[*\-\d.]+\s*", "", line)
                questions.append(line)

        logger.info(f"Parsed {len(questions)} related questions")
        return questions[:3]
    except Exception as e:
        logger.warning(f"Related questions generation failed: {e}")
        return []

# ======== Generator Logic ========

def stream_response(
    query: str,
    search_uuid: str,
    generate_related_questions: bool,
    client: openai.OpenAI,
    model_name: str,
) -> Generator[str, None, None]:
    """
    Main logic to:
    1. Retrieve context (RAG + Web)
    2. Stream LLM Answer
    3. Stream Related Questions
    4. Suggest Images
    5. Cache results
    """

    # 1. Retrieve Contexts
    contexts = []

    # RAG
    try:
        rag_contexts = get_rag_context(query)
        logger.info(f"RAG found {len(rag_contexts)} contexts")
        contexts.extend(rag_contexts)
    except Exception as e:
        logger.error(f"RAG error: {e}")

    # DuckDuckGo (fill up only if needed, to save time/tokens, or always add?)
    # Original logic: if len(contexts) < REFERENCE_COUNT
    if len(contexts) < REFERENCE_COUNT:
        try:
            web_results = search_with_duckduckgo(query)
            contexts.extend(web_results)
        except Exception as e:
            logger.error(f"Web search error: {e}")

    # Limit contexts
    contexts = contexts[:REFERENCE_COUNT]

    # Send Contexts to client
    yield json.dumps(contexts)
    yield "\n\n__LLM_RESPONSE__\n\n"

    # 2. Prepare LLM Prompt
    # Format context for prompt
    context_block = "\n\n".join(
        [f"[[citation:{i+1}]] {c['snippet']}" for i, c in enumerate(contexts)]
    )

    system_prompt = ecEasyPrompts._rag_query_text.format(context=context_block)

    llm_response_accumulated = []

    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            max_tokens=1024,
            stop=STOP_WORDS,
            stream=True,
            temperature=0.7,
        )

        for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                llm_response_accumulated.append(content)
                yield content

    except Exception as e:
        logger.error(f"LLM Stream error: {e}")
        yield f"\n[Error generating response: {e}]"

    # 3. Related Questions
    related_questions_json = "[]"
    if SHOULD_DO_RELATED_QUESTIONS and generate_related_questions:
        try:
            questions = get_related_questions(query, contexts, client, model_name)
            # Frontend expects keywords/questions in an object with "question" key
            formatted_questions = [{"question": q} for q in questions]
            related_questions_json = json.dumps(formatted_questions)
            yield "\n\n__RELATED_QUESTIONS__\n\n"
            yield related_questions_json
        except Exception as e:
            logger.error(f"Related questions error: {e}")

    # 4. Suggest Images
    suggested_images_json = "[]"
    if image_retriever is not None:
        try:
            llm_response_text = "".join(llm_response_accumulated)
            image_suggestions = suggest_images_for_response(query, llm_response_text, image_retriever)
            formatted_images = [
                {
                    "path": f"/ECEknowledge/{img['source_relpath']}",
                    "description": img.get("description", ""),
                    "doc_type": img.get("doc_type", "general"),
                    "source_relpath": img["source_relpath"]
                }
                for img in image_suggestions
            ]
            if formatted_images:
                suggested_images_json = json.dumps(formatted_images)
                logger.info(f"Suggested {len(formatted_images)} images for query")
                yield "\n\n__SUGGESTED_IMAGES__\n\n"
                yield suggested_images_json
        except Exception as e:
            logger.warning(f"Image suggestion error: {e}")

    # 5. Cache Result
    # We cache the full interaction for the "UUID" retrieval
    if search_uuid:
        full_response_data = [
            json.dumps(contexts),
            "\n\n__LLM_RESPONSE__\n\n",
            "".join(llm_response_accumulated),
            "\n\n__RELATED_QUESTIONS__\n\n" + related_questions_json
        ]
        if image_retriever is not None and suggested_images_json != "[]":
            full_response_data.extend(["\n\n__SUGGESTED_IMAGES__\n\n", suggested_images_json])
        
        try:
            with shelve.open(KV_NAME) as db:
                db[search_uuid] = full_response_data
        except Exception as e:
            logger.error(f"Cache write error: {e}")

# ======== FastAPI App ========

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize app-level resources on startup and release on shutdown."""
    global image_retriever
    if ImageRetriever is not None:
        try:
            image_retriever = ImageRetriever()
            num_images = len(image_retriever.get_all_images())
            logger.info(f"Image retriever initialized: {num_images} images available")
        except Exception as e:
            logger.warning(f"Failed to initialize image retriever: {e}. Image suggestions will be disabled.")
            image_retriever = None
    else:
        logger.warning("ImageRetriever not available. Image suggestions will be disabled.")
    yield


app = FastAPI(lifespan=lifespan)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        client, provider, model_name, using_server_key = resolve_runtime_llm_config(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve LLM runtime config: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize LLM provider")

    logger.info(
        f"Received query (provider={provider}, model={model_name}, key_source={'server' if using_server_key else 'user'})"
    )

    # Check cache first
    if request.search_uuid:
        try:
            with shelve.open(KV_NAME) as db:
                if request.search_uuid in db:
                    cached_data = db[request.search_uuid]
                    # cached_data is a list of strings (parts of the stream)
                    # We can stream it back
                    return StreamingResponse(iter(cached_data), media_type="text/plain")
        except Exception:
            pass

    return StreamingResponse(
        stream_response(
            request.query,
            request.search_uuid,
            bool(request.generate_related_questions),
            client,
            model_name,
        ),
        media_type="text/plain"
    )

@app.get("/")
def home():
    if UI_VERSION == "newui":
        return RedirectResponse("/newUI/index.html")
    if UI_VERSION == "frontpage":
        return RedirectResponse("/frontpage/index.html")
    return RedirectResponse("/ui/index.html")

@app.get("/frontpage")
def frontpage_redirect():
    return RedirectResponse("/frontpage/index.html")

@app.get("/frontpage/")
def frontpage_redirect_slash():
    return RedirectResponse("/frontpage/index.html")

# Mount static files
if os.path.exists("ui"):
    app.mount("/ui", StaticFiles(directory="ui"), name="ui")
if os.path.exists("newUI"):
    app.mount("/newUI", StaticFiles(directory="newUI"), name="newUI")
if os.path.exists(os.path.join("newDesign", "FrontPage")):
    app.mount("/frontpage", StaticFiles(directory=os.path.join("newDesign", "FrontPage")), name="frontpage")
if os.path.exists("ECEknowledge"):
    app.mount("/ECEknowledge", StaticFiles(directory="ECEknowledge"), name="eceknowledge")
if os.path.exists("localData"):
    app.mount("/localData", StaticFiles(directory="localData"), name="localData")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
