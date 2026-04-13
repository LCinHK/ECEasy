import httpx
import openai
from fastapi import HTTPException

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
DEEPSEEK_ALLOWED_MODELS = {"deepseek-r1", "deepseek-v3", "deepseek-v3-2-exp"}
DEFAULT_USER_MODEL_BY_PROVIDER = {
    "openai": "gpt-5-mini",
    "deepseek": "deepseek-v3",
}


def _resolve_provider_base_url(provider: str, request: QueryRequest, using_server_key: bool) -> str:
    if using_server_key:
        return OPENAI_BASE_URL if provider == "openai" else DEEPSEEK_BASE_URL

    requested_base_url = request.base_url
    if requested_base_url is not None:
        return str(requested_base_url).rstrip("/")

    return OPENAI_BASE_URL if provider == "openai" else DEEPSEEK_BASE_URL


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

    return model_name


def resolve_runtime_llm_config(request: QueryRequest) -> tuple[openai.OpenAI, str, str, bool]:
    provider = (request.llm_provider or LLM_PROVIDER).lower()
    if provider not in {"ollama", "openai", "deepseek"}:
        raise HTTPException(status_code=400, detail="llm_provider must be one of: ollama, openai, deepseek")

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

    if provider == "openai":
        api_key = OPENAI_API_KEY if using_server_key else user_api_key
        if not api_key:
            raise HTTPException(status_code=400, detail="OpenAI API key is required for the selected mode.")
        client = openai.OpenAI(api_key=api_key, base_url=_resolve_provider_base_url(provider, request, using_server_key))
    else:
        api_key = DEEPSEEK_API_KEY if using_server_key else user_api_key
        if not api_key:
            raise HTTPException(status_code=400, detail="DeepSeek API key is required for the selected mode.")
        client = openai.OpenAI(api_key=api_key, base_url=_resolve_provider_base_url(provider, request, using_server_key))

    model_name = _resolve_remote_model_name(provider, request, using_server_key)
    return client, provider, model_name, using_server_key
