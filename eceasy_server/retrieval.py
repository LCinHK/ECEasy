import json
import re
import uuid
from typing import List

import openai
from loguru import logger

import ecEasyPrompts
from .config import KNOWLEDGE, REFERENCE_COUNT

try:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
except ImportError:
    logger.warning("duckduckgo_search / ddgs not installed. Web search will be disabled.")
    DDGS = None


if KNOWLEDGE == "faiss":
    try:
        from faiss_rag import get_rag_context
        logger.info("Knowledge base: FAISS (ECE knowledge — faiss)")
    except ImportError as e:
        logger.warning(f"Could not import faiss_rag: {e}. RAG functionality will be disabled.")

        def get_rag_context(_: str) -> list:
            return []
else:
    try:
        from arag.arag import get_rag_context
        logger.info("Knowledge base: ChromaDB (Network knowledge — arag/chromaVectorStore/)")
    except ImportError as e:
        logger.warning(f"Could not import arag.arag: {e}. RAG functionality will be disabled.")

        def get_rag_context(_: str) -> list:
            return []


def search_with_duckduckgo(query: str) -> List[dict]:
    if not DDGS:
        return []

    try:
        results = []
        for attempt in range(3):
            try:
                with DDGS() as ddgs:
                    ddgs_gen = ddgs.text(query, max_results=REFERENCE_COUNT)
                    if ddgs_gen:
                        results = list(ddgs_gen)
                        if results:
                            break
            except Exception as e:
                logger.warning(f"DuckDuckGo attempt {attempt + 1} failed: {e}")

        logger.info(f"DuckDuckGo found {len(results)} results")

        return [
            {
                "id": str(uuid.uuid4()),
                "name": r.get("title", "Source"),
                "url": r.get("href", "#"),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return []


def _clean_related_question(text: str) -> str:
    cleaned = text.strip().strip('"').strip("'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _parse_related_questions(content: str) -> List[str]:
    if not content:
        return []

    # 1) Preferred format: JSON array of strings.
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            out = []
            for item in parsed:
                if isinstance(item, str):
                    q = _clean_related_question(item)
                    if q:
                        out.append(q)
            if out:
                return out
    except Exception:
        pass

    # 2) Common fallback: markdown / plain text list lines.
    questions: List[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Drop markdown bullets/numbering prefixes.
        line = re.sub(r"^(?:[*\-•]|\d+[.)])\s*", "", line)
        line = _clean_related_question(line)

        # Accept questions with either ASCII or full-width question mark,
        # and also accept imperative follow-up forms.
        if line.endswith("?") or line.endswith("？") or len(line) > 8:
            questions.append(line)

    # 3) If model returned a quasi-JSON single line that failed loads,
    # extract quoted strings as a last resort.
    if not questions:
        quoted_items = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', content)
        for item in quoted_items:
            q = _clean_related_question(item)
            if q:
                questions.append(q)

    # Preserve order while de-duplicating.
    deduped: List[str] = []
    seen = set()
    for q in questions:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(q)

    return deduped


def get_related_questions(
    query: str,
    contexts: List[dict],
    client: openai.OpenAI,
    model_name: str,
) -> List[str]:
    if not contexts:
        return []

    context_text = "\n\n".join([c["snippet"] for c in contexts])[:4000]
    prompt = ecEasyPrompts._more_questions_prompt.format(context=context_text)
    prompt += f"\n{query}"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],  # type: ignore[arg-type]
            max_tokens=256,
            temperature=0.7,
        )
        if not response.choices:
            return []

        content = response.choices[0].message.content
        logger.info(f"Related questions raw output: {content}")

        questions = _parse_related_questions(content)
        logger.info(f"Parsed {len(questions)} related questions")
        return questions[:3]
    except Exception as e:
        logger.warning(f"Related questions generation failed: {e}")
        return []
