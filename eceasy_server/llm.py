import httpx
import openai
from fastapi import HTTPException
import os

from .config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    get_model_name_for_provider,
)
from .schemas import QueryRequest

OPENAI_ALLOWED_MODELS = {
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-4o",
    "gpt-4.1",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-5-mini",
    "gpt-5-nano",
}
<<<<<<< Updated upstream
DEEPSEEK_ALLOWED_MODELS = {"deepseek-r1", "deepseek-v3", "deepseek-v3-2-exp"}
DEFAULT_USER_MODEL_BY_PROVIDER = {
    "openai": "gpt-5-mini",
    "deepseek": "deepseek-v3",
=======
DEEPSEEK_ALLOWED_MODELS = {"deepseek-chat","deepseek-reasoner"}

GROK_ALLOWED_MODELS = {
    "grok-4.3",
    "grok-4-1-fast-reasoning",
    "grok-4-fast-reasoning",
    "grok-4-1-fast-non-reasoning",
    "grok-4-fast-non-reasoning",
    "grok-3",
    "grok-3-mini",
    "grok-beta"
}

DEFAULT_USER_MODEL_BY_PROVIDER = {
    "openai": "gpt-5-mini",
    "deepseek": "deepseek-chat",
    "grok": "grok-4.3",
>>>>>>> Stashed changes
}


def _resolve_remote_model_name(provider: str, request: QueryRequest, using_server_key: bool) -> str:
    if using_server_key:
        # Server-key mode is always controlled by backend env config.
        return get_model_name_for_provider(provider)

    requested_model = (request.llm_model or "").strip()
    model_name = requested_model or DEFAULT_USER_MODEL_BY_PROVIDER.get(provider, "")

    if provider == "openai" and model_name not in OPENAI_ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported OpenAI model selected.")
    if provider == "deepseek" and model_name not in DEEPSEEK_ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported DeepSeek model selected.")
    if provider == "grok" and model_name not in GROK_ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"Model not found: {model_name}")


def resolve_runtime_llm_config(request: QueryRequest) -> tuple[openai.OpenAI, str, str, bool]:
    provider = (request.llm_provider or LLM_PROVIDER).lower()

    # ← Updated to accept grok
    if provider not in {"ollama", "openai", "deepseek", "grok"}:
        raise HTTPException(status_code=400, detail="llm_provider must be one of: ollama, openai, deepseek, grok")

    if provider == "ollama":
        client = openai.OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
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

    if legacy_client_mode:
        using_server_key = True

    if not user_api_key and not using_server_key:
        raise HTTPException(
            status_code=400,
            detail="Provide api_key or set use_server_key=true for remote providers.",
        )

    # ← GROK SUPPORT ADDED HERE
    if provider == "grok":
        api_key = os.getenv("GROK_API_KEY") if using_server_key else user_api_key
        if not api_key:
            raise HTTPException(status_code=400, detail="Grok API key is required for the selected mode.")
        client = openai.OpenAI(
            api_key=api_key,
            base_url=os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
        )

    elif provider == "openai":
        api_key = OPENAI_API_KEY if using_server_key else user_api_key
        if not api_key:
            raise HTTPException(status_code=400, detail="OpenAI API key is required for the selected mode.")
<<<<<<< Updated upstream
        client = openai.OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
    else:
=======
        client = openai.OpenAI(api_key=api_key, base_url=_resolve_provider_base_url(provider, request, using_server_key))

    else:  # deepseek
>>>>>>> Stashed changes
        api_key = DEEPSEEK_API_KEY if using_server_key else user_api_key
        if not api_key:
            raise HTTPException(status_code=400, detail="DeepSeek API key is required for the selected mode.")
        client = openai.OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    model_name = _resolve_remote_model_name(provider, request, using_server_key)
    return client, provider, model_name, using_server_key