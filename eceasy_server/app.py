import logging
import os
import shelve
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from .config import KV_NAME, UI_VERSION
from .schemas import QueryRequest
from .services import resolve_runtime_llm_config, stream_response

# Suppress noisy third-party loggers.
logging.getLogger("chromadb").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

try:
    from image_retrieval import ImageRetriever, suggest_images_for_response
except ImportError:
    logger.warning("image_retrieval module not found. Image suggestions will be disabled.")
    ImageRetriever = None
    suggest_images_for_response = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.image_retriever = None
    if ImageRetriever is not None:
        try:
            app.state.image_retriever = ImageRetriever()
            num_images = len(app.state.image_retriever.get_all_images())
            logger.info(f"Image retriever initialized: {num_images} images available")
        except Exception as e:
            logger.warning(f"Failed to initialize image retriever: {e}. Image suggestions will be disabled.")
            app.state.image_retriever = None
    else:
        logger.warning("ImageRetriever not available. Image suggestions will be disabled.")
    yield


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError):
        logger.error(f"Validation error: {exc.errors()}")
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

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

        if request.search_uuid:
            try:
                with shelve.open(KV_NAME) as db:
                    if request.search_uuid in db:
                        return StreamingResponse(iter(db[request.search_uuid]), media_type="text/plain")
            except Exception:
                pass

        return StreamingResponse(
            stream_response(
                request.query,
                request.search_uuid,
                bool(request.generate_related_questions),
                client,
                model_name,
                image_retriever=app.state.image_retriever,
                image_suggester=suggest_images_for_response,
            ),
            media_type="text/plain",
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

    return app


app = create_app()

