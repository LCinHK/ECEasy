import os
import warnings

from dotenv import load_dotenv
from loguru import logger

# Disable ChromaDB telemetry noise.
os.environ["ANONYMIZED_TELEMETRY"] = "False"

warnings.filterwarnings(
    "ignore",
    message=".*Accessing the 'model_fields' attribute on the instance is deprecated.*",
)

load_dotenv(override=True)

# --- Server Config ---
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))

# --- Knowledge Base Selection ---
KNOWLEDGE = os.environ.get("KNOWLEDGE", "faiss").lower()

# --- UI Version ---
UI_VERSION = os.environ.get("UI_VERSION", "newUI").lower()

# --- LLM Provider ---
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()

# --- Feature Flags ---
KV_NAME = os.environ.get("KV_NAME", "eceasy-chat-local.kv")
REFERENCE_COUNT_MIN = int(os.environ.get("REFERENCE_COUNT_MIN", "4"))
REFERENCE_COUNT_MAX = int(os.environ.get("REFERENCE_COUNT_MAX", "10"))
REFERENCE_COUNT_DEFAULT = int(
    os.environ.get("REFERENCE_COUNT_DEFAULT", str((REFERENCE_COUNT_MIN + REFERENCE_COUNT_MAX) // 2))
)
SHOULD_DO_RELATED_QUESTIONS = os.environ.get("RELATED_QUESTIONS", "true").lower() == "true"

# --- Provider Specific Config ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", os.environ.get("LLM_REMOTE_OPENAI_API_KEY", ""))
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", os.environ.get("LLM_REMOTE_OPENAI_MODEL", "gpt-4o"))
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", os.environ.get("LLM_REMOTE_OPENAI_URL", "https://api.openai.com/v1"))

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", os.environ.get("LLM_REMOTE_API_KEY", ""))
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", os.environ.get("LLM_REMOTE_URL", "https://api.deepseek.com"))
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", os.environ.get("LLM_REMOTE_MODEL", "deepseek-chat"))

# === GROK CONFIG ===
GROK_API_KEY = os.environ.get("GROK_API_KEY", "")
GROK_MODEL = os.environ.get("GROK_MODEL", "grok-4.3")
GROK_BASE_URL = os.environ.get("GROK_BASE_URL", "https://api.x.ai/v1")

STOP_WORDS = [
    "<|im_end|>",
    "[End]",
    "\nReferences:\n",
    "\nSources:\n",
]


def resolve_reference_count(query: str) -> int:
    default_count = max(REFERENCE_COUNT_MIN, min(REFERENCE_COUNT_MAX, REFERENCE_COUNT_DEFAULT))

    text = (query or "").strip()
    if not text:
        return default_count

    score_delta = 0
    lowered = text.lower()

    # Longer, multi-part questions usually benefit from broader context retrieval.
    if len(text) >= 90:
        score_delta += 2
    elif len(text) >= 45:
        score_delta += 1
    elif len(text) <= 18:
        score_delta -= 2
    elif len(text) <= 32:
        score_delta -= 1

    complexity_keywords = (
        "compare",
        "difference",
        "plan",
        "pathway",
        "roadmap",
        "requirements",
        "prerequisite",
        "elective",
        "internship",
        "exchange",
        "fyp",
        "thesis",
    )
    keyword_hits = sum(1 for kw in complexity_keywords if kw in lowered)
    if keyword_hits >= 3:
        score_delta += 2
    elif keyword_hits >= 1:
        score_delta += 1

    # Multi-question prompts tend to need more evidence chunks.
    question_marks = text.count("?") + text.count("？")
    if question_marks >= 2:
        score_delta += 1

    target = default_count + score_delta
    return max(REFERENCE_COUNT_MIN, min(REFERENCE_COUNT_MAX, target))


def get_model_name_for_provider(provider: str) -> str:
    if provider == "openai":
        return OPENAI_MODEL
    if provider == "deepseek":
        return DEEPSEEK_MODEL
    if provider == "grok":
        return GROK_MODEL
    return OLLAMA_MODEL


LLM_MODEL = get_model_name_for_provider(LLM_PROVIDER)

logger.info(f"Knowledge base : {KNOWLEDGE.upper()}")
logger.info(f"UI Version     : {UI_VERSION}")
logger.info(f"LLM Provider   : {LLM_PROVIDER}, Model: {LLM_MODEL}")
logger.info(f"Server         : {HOST}:{PORT}")

