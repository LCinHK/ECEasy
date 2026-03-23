"""Compatibility facade for split service modules.

Existing imports from `eceasy_server.services` remain valid while core logic
now lives in dedicated modules:
- `eceasy_server.llm`
- `eceasy_server.retrieval`
- `eceasy_server.streaming`
"""

from .llm import resolve_runtime_llm_config
from .retrieval import get_rag_context, get_related_questions, search_with_duckduckgo
from .streaming import stream_response

__all__ = [
    "resolve_runtime_llm_config",
    "get_rag_context",
    "get_related_questions",
    "search_with_duckduckgo",
    "stream_response",
]
