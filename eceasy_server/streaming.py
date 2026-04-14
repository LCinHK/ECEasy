import json
import shelve
import ast
import re
from typing import Callable, Generator, List, Optional
from urllib.parse import quote

import openai
from loguru import logger

import ecEasyPrompts
from .config import KV_NAME, SHOULD_DO_RELATED_QUESTIONS, STOP_WORDS, resolve_reference_count
from .retrieval import get_rag_context_with_limit, get_related_questions, search_with_duckduckgo

SERVER_FIXED_MEMORY_TURNS = 3
MAX_MEMORY_TURNS = 15
SERVER_PROMPT_TOKEN_BUDGET = 3200


def _sanitize_error_text(text: str) -> str:
    if not text:
        return ""
    sanitized = re.sub(r"sk-[A-Za-z0-9\-_]+", "sk-***", text)
    sanitized = re.sub(r"(ApiKey\s*[:：]\s*)([^\s)\]，,]+)", r"\1***", sanitized, flags=re.IGNORECASE)
    return sanitized.strip()


def _extract_error_payload(raw_error: str) -> dict:
    if not raw_error:
        return {}

    start = raw_error.find("{")
    end = raw_error.rfind("}")
    if start < 0 or end <= start:
        return {}

    payload_text = raw_error[start:end + 1]
    try:
        parsed = ast.literal_eval(payload_text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _parse_llm_error(exc: Exception) -> tuple[int | None, str, str]:
    raw = str(exc)
    status_code = getattr(exc, "status_code", None)
    code = ""
    message = ""

    if isinstance(status_code, str) and status_code.isdigit():
        status_code = int(status_code)

    if status_code is None:
        m = re.search(r"Error\s+code\s*:\s*(\d{3})", raw, flags=re.IGNORECASE)
        if m:
            status_code = int(m.group(1))

    payload = _extract_error_payload(raw)
    if payload:
        err_obj = payload.get("error", payload)
        if isinstance(err_obj, dict):
            message = str(err_obj.get("message") or "").strip()
            code = str(err_obj.get("code") or err_obj.get("type") or "").strip()

    if not message:
        message = raw

    return status_code, _sanitize_error_text(message), _sanitize_error_text(code)


def _format_llm_error_for_user(exc: Exception) -> str:
    status_code, message, code = _parse_llm_error(exc)
    lower = message.lower()

    category = "unknown"
    user_msg = "The language model request failed. Please try again later."

    if status_code in {401, 403} or "forbidden" in lower or "unauthorized" in lower:
        category = "auth_or_permission"
        user_msg = "The API key or endpoint does not have permission for this request."
    if "token" in lower and ("4096" in lower or "prompt" in lower or "maximum context" in lower or "too long" in lower):
        category = "prompt_too_long"
        user_msg = "The request is too long for this API key/endpoint token limit. Try a shorter question or reduce memory turns."
    elif status_code == 429 or "rate limit" in lower or "too many requests" in lower:
        category = "rate_limited"
        user_msg = "Rate limit reached. Please wait and retry."
    elif status_code in {500, 502, 503, 504} or "timeout" in lower:
        category = "provider_unavailable"
        user_msg = "The model provider is temporarily unavailable. Please retry in a moment."

    detail_parts = []
    if status_code is not None:
        detail_parts.append(f"status={status_code}")
    if code:
        detail_parts.append(f"code={code}")

    detail_suffix = f" ({', '.join(detail_parts)})" if detail_parts else ""
    excerpt = f" Provider says: {message[:220]}" if message else ""
    return f"\n[Error generating response: {user_msg}{detail_suffix}.{excerpt}]"


def _clamp_memory_turns(memory_turns: int, using_server_key: bool) -> int:
    if using_server_key:
        return SERVER_FIXED_MEMORY_TURNS
    return max(0, min(MAX_MEMORY_TURNS, int(memory_turns)))


def _build_windowed_history(conversation_history: List[dict], memory_turns: int) -> List[dict]:
    if memory_turns <= 0:
        return []

    cleaned: List[dict] = []
    for turn in conversation_history:
        role = str(turn.get("role", "")).strip()
        content = str(turn.get("content", "")).strip()
        if role not in ("user", "assistant") or not content:
            continue
        cleaned.append({"role": role, "content": content[:4000]})

    # Treat one "turn" as one user+assistant pair, i.e., at most 2 messages per turn.
    return cleaned[-(memory_turns * 2):]


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text or "") // 4) + 1)


def _reduce_history_for_budget(
    system_prompt: str,
    query: str,
    conversation_history: List[dict],
    memory_turns: int,
    prompt_budget: int,
) -> tuple[int, List[dict], int]:
    effective_memory_turns = max(0, int(memory_turns))
    history_for_prompt = _build_windowed_history(conversation_history, effective_memory_turns)

    def estimated_total(history: List[dict]) -> int:
        history_tokens = sum(_estimate_tokens(turn.get("content", "")) for turn in history)
        return _estimate_tokens(system_prompt) + _estimate_tokens(query) + history_tokens

    total_tokens = estimated_total(history_for_prompt)
    while effective_memory_turns > 0 and total_tokens > prompt_budget:
        effective_memory_turns -= 1
        history_for_prompt = _build_windowed_history(conversation_history, effective_memory_turns)
        total_tokens = estimated_total(history_for_prompt)

    return effective_memory_turns, history_for_prompt, total_tokens


def stream_response(
    query: str,
    search_uuid: str,
    generate_related_questions: bool,
    client: openai.OpenAI,
    model_name: str,
    conversation_history: Optional[List[dict]] = None,
    memory_turns: int = SERVER_FIXED_MEMORY_TURNS,
    using_server_key: bool = True,
    image_retriever: Optional[object] = None,
    image_suggester: Optional[Callable[[str, str, object], List[dict]]] = None,
) -> Generator[str, None, None]:
    reference_count = resolve_reference_count(query)
    contexts = []

    try:
        rag_contexts = get_rag_context_with_limit(query, reference_count)
        logger.info(f"RAG found {len(rag_contexts)} contexts")
        contexts.extend(rag_contexts)
    except Exception as e:
        logger.error(f"RAG error: {e}")

    if len(contexts) < reference_count:
        try:
            web_results = search_with_duckduckgo(query, max_results=reference_count)
            contexts.extend(web_results)
        except Exception as e:
            logger.error(f"Web search error: {e}")

    contexts = contexts[:reference_count]

    yield json.dumps(contexts)
    yield "\n\n__LLM_RESPONSE__\n\n"

    context_block = "\n\n".join([f"[[citation:{i + 1}]] {c['snippet']}" for i, c in enumerate(contexts)])
    system_prompt = ecEasyPrompts._rag_query_text.format(context=context_block)

    effective_memory_turns = _clamp_memory_turns(memory_turns, using_server_key)
    history_for_prompt = _build_windowed_history(conversation_history or [], effective_memory_turns)
    if using_server_key:
        effective_memory_turns, history_for_prompt, estimated_tokens = _reduce_history_for_budget(
            system_prompt,
            query,
            conversation_history or [],
            effective_memory_turns,
            SERVER_PROMPT_TOKEN_BUDGET,
        )
        if estimated_tokens > SERVER_PROMPT_TOKEN_BUDGET:
            logger.warning(
                f"Conversation prompt still estimated at ~{estimated_tokens} tokens after trimming memory; the provider may still reject it."
            )
    logger.info(
        f"Conversation memory active: {effective_memory_turns} turn(s), using {len(history_for_prompt)} prior message(s)"
    )

    llm_response_accumulated = []

    try:
        chat_messages = [
            {"role": "system", "content": system_prompt},
            *history_for_prompt,
            {"role": "user", "content": query},
        ]

        stream = client.chat.completions.create(
            model=model_name,
            messages=chat_messages,
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
        parsed_status, parsed_msg, parsed_code = _parse_llm_error(e)
        logger.error(
            f"LLM Stream error parsed (status={parsed_status}, code={parsed_code}): {parsed_msg}"
        )
        yield _format_llm_error_for_user(e)

    related_questions_json = "[]"
    if SHOULD_DO_RELATED_QUESTIONS and generate_related_questions:
        try:
            questions = get_related_questions(query, contexts, client, model_name)
            related_questions_json = json.dumps([{"question": q} for q in questions])
            yield "\n\n__RELATED_QUESTIONS__\n\n"
            yield related_questions_json
        except Exception as e:
            logger.error(f"Related questions error: {e}")

    suggested_images_json = "[]"
    if image_retriever is not None and image_suggester is not None:
        try:
            llm_response_text = "".join(llm_response_accumulated)
            image_suggestions = image_suggester(query, llm_response_text, image_retriever)
            formatted_images = [
                {
                    "path": _to_resource_url("ECEknowledge", img["source_relpath"]),
                    "description": img.get("description", ""),
                    "doc_type": img.get("doc_type", "general"),
                    "source_relpath": img["source_relpath"],
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

    if search_uuid:
        full_response_data = [
            json.dumps(contexts),
            "\n\n__LLM_RESPONSE__\n\n",
            "".join(llm_response_accumulated),
            "\n\n__RELATED_QUESTIONS__\n\n" + related_questions_json,
        ]
        if image_retriever is not None and suggested_images_json != "[]":
            full_response_data.extend(["\n\n__SUGGESTED_IMAGES__\n\n", suggested_images_json])

        try:
            with shelve.open(KV_NAME) as db:
                db[search_uuid] = full_response_data
        except Exception as e:
            logger.error(f"Cache write error: {e}")

def _to_resource_url(root: str, relpath: str) -> str:
    safe_rel = quote(relpath.replace("\\", "/").lstrip("/"), safe="/")
    return f"/resource/{root}/{safe_rel}" if safe_rel else "#"
