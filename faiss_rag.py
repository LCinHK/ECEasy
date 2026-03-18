"""
FAISS-based RAG module for ECEasy.
Uses the FAISS index built from ECEknowledge/ (by ingest_university.py).
The embedding model and index path are both controlled via .env.
"""

import os
import re
import logging

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

# ======== Similarity score threshold ========
# FAISS returns L2 distance; lower = more similar.
# Chunks with distance >= this value are considered irrelevant and filtered out.
FAISS_SCORE_THRESHOLD = 1.5

# Retrieve a larger pool first, then rerank with metadata/code-aware boosts.
FAISS_CANDIDATE_K = int(os.environ.get("FAISS_CANDIDATE_K", "40"))
FAISS_FINAL_K = int(os.environ.get("FAISS_FINAL_K", "8"))

COURSE_CODE_RE = re.compile(r"\b([A-Za-z]{4})\s*[-_]?\s*(\d{4}[A-Za-z]?)\b")


def _normalize_course_code(text: str) -> str:
    compact = re.sub(r"\s+", "", text).replace("-", "").replace("_", "").upper()
    m = COURSE_CODE_RE.search(compact)
    if not m:
        return ""
    return f"{m.group(1).upper()}{m.group(2).upper()}"


def _extract_course_code(text: str) -> str:
    m = COURSE_CODE_RE.search(text)
    if not m:
        return ""
    return f"{m.group(1).upper()}{m.group(2).upper()}"


# ======== Embedding model + index path resolution ========

def _index_name_from_hub(hub_name: str) -> str:
    """
    Derives a filesystem-safe FAISS index folder name from a Hub model name.
      "BAAI/bge-small-en-v1.5"  →  "faiss_index_bge-small-en-v1.5"
      "all-MiniLM-L6-v2"        →  "faiss_index_all-MiniLM-L6-v2"
    Uses only the last path component (after any '/') so org prefixes are stripped.
    """
    short = hub_name.split("/")[-1]
    return f"faiss_index_{short}"


def _resolve_embedding_model() -> tuple[str, str]:
    """
    Returns (model_name_or_path, faiss_index_path).

    Model resolution priority:
      1. If EMBEDDING_MODEL_LOCAL_PATH in .env points to an existing local directory
         → use it directly (fully offline; sets TRANSFORMERS_OFFLINE=1).
      2. Otherwise use EMBEDDING_MODEL_HUB_NAME as a Hub ID for auto-download/cache.

    The FAISS index folder is always derived from EMBEDDING_MODEL_HUB_NAME so that
    different models store their indexes in separate directories and never overwrite
    each other. If EMBEDDING_MODEL_HUB_NAME is not set, falls back to "all-MiniLM-L6-v2".
    """
    hub_name = os.environ.get("EMBEDDING_MODEL_HUB_NAME", "all-MiniLM-L6-v2").strip()
    index_path = _index_name_from_hub(hub_name)

    local_path = os.environ.get("EMBEDDING_MODEL_LOCAL_PATH", "").strip()
    if local_path:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        resolved = os.path.normpath(os.path.join(base_dir, local_path))
        if os.path.isdir(resolved):
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            print(f"[INFO] Embedding model : local folder '{resolved}' (offline mode)")
            print(f"[INFO] FAISS index path: '{index_path}'")
            logger.info(f"[FAISS RAG] Local model '{resolved}', index '{index_path}'")
            return resolved, index_path
        else:
            print(f"[WARNING] EMBEDDING_MODEL_LOCAL_PATH '{local_path}' (resolved: '{resolved}') "
                  f"not found — falling back to HuggingFace Hub.")
            logger.warning("[FAISS RAG] Local model path not found, falling back to Hub.")

    print(f"[INFO] Embedding model : HuggingFace Hub '{hub_name}' (requires internet on first run)")
    print(f"[INFO] FAISS index path: '{index_path}'")
    logger.info(f"[FAISS RAG] Hub model '{hub_name}', index '{index_path}'")
    return hub_name, index_path


# ======== Load embedding model & vector store once at module import ========
_model_name, FAISS_INDEX_PATH = _resolve_embedding_model()

_embeddings = HuggingFaceEmbeddings(
    model_name=_model_name,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

_vectorstore = None
try:
    _vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings=_embeddings,
        allow_dangerous_deserialization=True,  # Required for local pickled index
    )
    logger.info(f"[FAISS RAG] Index loaded successfully from '{FAISS_INDEX_PATH}'")
    print(f"[INFO] FAISS RAG index loaded from '{FAISS_INDEX_PATH}'")
except Exception as e:
    logger.warning(f"[FAISS RAG] Could not load FAISS index: {e}. FAISS RAG will be disabled.")
    print(f"[WARNING] Could not load FAISS index: {e}. FAISS RAG will be disabled.")


def get_rag_context(query: str):
    """
    Retrieve relevant document chunks from the FAISS index for the given query.
    Returns a list of context dicts compatible with the server's streaming pipeline:
        [{ 'name': str, 'snippet': str, 'url': str }, ...]
    """
    if _vectorstore is None:
        return []

    query_course_code = _extract_course_code(query)
    query_course_code = _normalize_course_code(query_course_code) if query_course_code else ""

    try:
        # similarity_search_with_score returns (Document, score) tuples
        retrieved = _vectorstore.similarity_search_with_score(query, k=max(FAISS_CANDIDATE_K, FAISS_FINAL_K))
    except Exception as e:
        logger.error(f"[FAISS RAG] Search failed: {e}")
        return []

    reranked = []
    for doc, raw_score in retrieved:
        metadata = doc.metadata or {}
        file_path = metadata.get("source", metadata.get("file_path", ""))
        doc_course_code = _normalize_course_code(str(metadata.get("course_code", "")))
        doc_department = str(metadata.get("department", "")).upper()

        score = float(raw_score)
        text_window = f"{doc.page_content[:1200]}\n{file_path}"

        if query_course_code:
            if doc_course_code and doc_course_code == query_course_code:
                score -= 0.45
            elif query_course_code in text_window.upper():
                score -= 0.25

            if doc_course_code and doc_course_code != query_course_code:
                score += 0.12

            if doc_department and query_course_code.startswith(doc_department):
                score -= 0.05

        reranked.append((doc, raw_score, score))

    reranked.sort(key=lambda x: x[2])

    threshold = 1.2 if query_course_code else FAISS_SCORE_THRESHOLD

    # Debug
    print(f"[FAISS RAG Debug] Query: {query}")
    if query_course_code:
        print(f"[FAISS RAG Debug] Detected course code: {query_course_code}")
    for doc, raw_score, rerank_score in reranked[:FAISS_FINAL_K]:
        meta = doc.metadata or {}
        cc = meta.get("course_code", "")
        print(f"[FAISS RAG Debug] raw={raw_score:.4f}, rerank={rerank_score:.4f}, cc={cc} | {doc.page_content[:60]}...")

    context = []
    for doc, _, score in reranked[:FAISS_FINAL_K]:
        if score >= threshold:
            continue  # Too dissimilar — skip

        metadata = doc.metadata
        # LangChain PDF loaders store the source path in 'source'
        file_path = metadata.get("source", metadata.get("file_path", ""))

        if "page" in metadata:
            # page is 0-indexed in LangChain loaders
            page_num = int(metadata["page"]) + 1
            name = f"Page {page_num}, {os.path.basename(file_path)}"
            url_suffix = f"#page={page_num}"
        else:
            name = os.path.basename(file_path) if file_path else "Source"
            url_suffix = ""

        course_code = _normalize_course_code(str(metadata.get("course_code", "")))
        doc_type = metadata.get("doc_type", "")
        if course_code:
            name = f"[{course_code}] {name}"
        if doc_type:
            name = f"{name} ({doc_type})"

        # Normalise path separators and strip leading ../
        clean_url_path = re.sub(r"\.\.", "", re.sub(r"\\+", "/", file_path))

        context.append({
            "name": name,
            "snippet": doc.page_content,
            "url": clean_url_path + url_suffix,
        })

    # De-duplicate by snippet content
    unique_context = list({entry["snippet"]: entry for entry in context}.values())

    return unique_context

