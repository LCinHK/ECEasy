import json
import shelve
from typing import Callable, Generator, List, Optional
from urllib.parse import quote

import openai
from loguru import logger

import ecEasyPrompts
from .config import KV_NAME, REFERENCE_COUNT, SHOULD_DO_RELATED_QUESTIONS, STOP_WORDS
from .retrieval import get_rag_context, get_related_questions, search_with_duckduckgo

SERVER_FIXED_MEMORY_TURNS = 3
MAX_MEMORY_TURNS = 15


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
    contexts = []

    try:
        rag_contexts = get_rag_context(query)
        logger.info(f"RAG found {len(rag_contexts)} contexts")
        contexts.extend(rag_contexts)
    except Exception as e:
        logger.error(f"RAG error: {e}")

    if len(contexts) < REFERENCE_COUNT:
        try:
            web_results = search_with_duckduckgo(query)
            contexts.extend(web_results)
        except Exception as e:
            logger.error(f"Web search error: {e}")

    contexts = contexts[:REFERENCE_COUNT]

    yield json.dumps(contexts)
    yield "\n\n__LLM_RESPONSE__\n\n"

    context_block = "\n\n".join([f"[[citation:{i + 1}]] {c['snippet']}" for i, c in enumerate(contexts)])
    system_prompt = ecEasyPrompts._rag_query_text.format(context=context_block)

    effective_memory_turns = _clamp_memory_turns(memory_turns, using_server_key)
    history_for_prompt = _build_windowed_history(conversation_history or [], effective_memory_turns)
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
            stop=STOP_WORDS if "grok" not in model_name.lower() else None,
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
