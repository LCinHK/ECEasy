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
        client = openai.OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
    else:
        api_key = DEEPSEEK_API_KEY if using_server_key else user_api_key
        if not api_key:
            raise HTTPException(status_code=400, detail="DeepSeek API key is required for the selected mode.")
        client = openai.OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    return client, provider, get_model_name_for_provider(provider), using_server_key

