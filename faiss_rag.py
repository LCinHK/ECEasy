"""
FAISS-based RAG module for ECEasy.
Uses the FAISS index built from ECEknowledge/ (by ingest_FAISS.py).
The embedding model and index path are both controlled via .env.
"""

import os
import re
import logging
from pathlib import Path
from urllib.parse import quote

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

COURSE_CODE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{4})\s*[-_]?\s*(\d{4}[A-Za-z]?)(?![A-Za-z0-9])")


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


def _infer_source_relpath(file_path: str, metadata: dict) -> str:
    """Best-effort relative path under ECEknowledge for old/new indexes."""
    rel = str(metadata.get("source_relpath", "")).strip()
    if rel:
        rel = re.sub(r"\\+", "/", rel).lstrip("/")
        if rel.lower().startswith("eceknowledge/"):
            rel = rel.split("/", 1)[1] if "/" in rel else ""
        return rel

    normalized = re.sub(r"\\+", "/", str(file_path or "")).strip()
    if not normalized:
        return ""

    lower_norm = normalized.lower()
    marker = "eceknowledge/"
    idx = lower_norm.find(marker)
    if idx >= 0:
        return normalized[idx + len(marker):].lstrip("/")

    return os.path.basename(normalized)


def _safe_filename_from_path(file_path: str) -> str:
    """Return filename only, handling both Unix and Windows separators."""
    normalized = re.sub(r"\\+", "/", str(file_path or "")).strip()
    if not normalized:
        return "Source"
    return normalized.split("/")[-1] or "Source"


def _collect_official_course_codes() -> set[str]:
    """Collect course codes from concrete course file names for validation/reranking."""
    base = Path(__file__).resolve().parent / "ECEknowledge" / "course_syllabus"
    if not base.exists():
        return set()

    codes: set[str] = set()
    # 1) Standalone course files.
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        code = _extract_course_code(p.stem)
        if code:
            codes.add(_normalize_course_code(code))

    # 2) The official ELEC syllabus aggregator contains many current course headings.
    elec_docx = base / "ELEC" / "ELEC_Syllabus_25-26_fall_spring.docx"
    if elec_docx.exists():
        try:
            from docx2txt import process as docx_process

            text = docx_process(str(elec_docx))
            for m in re.finditer(r"#\s*([A-Za-z]{4}\s*[-_]?\s*\d{4}[A-Za-z]?)\s*-", text):
                code = _normalize_course_code(m.group(1))
                if code:
                    codes.add(code)
        except Exception as e:
            logger.warning(f"[FAISS RAG] Could not scan official syllabus docx for course codes: {e}")

    return codes


def _lookup_official_course_fact(course_code: str) -> dict | None:
    """Derive an authoritative course fact from official syllabus source text."""
    code = _normalize_course_code(course_code)
    if not code:
        return None

    base = Path(__file__).resolve().parent / "ECEknowledge" / "course_syllabus"
    candidates = [
        base / "ELEC" / "ELEC_Syllabus_25-26_fall_spring.docx",
    ]

    # Also consider standalone files if present in future data updates.
    candidates.extend(sorted(base.rglob(f"*{code}*.*")))

    for path in candidates:
        if not path.exists():
            continue

        text = ""
        try:
            if path.suffix.lower() == ".docx":
                from docx2txt import process as docx_process

                text = docx_process(str(path))
            elif path.suffix.lower() == ".pdf":
                try:
                    from pypdf import PdfReader

                    reader = PdfReader(str(path))
                    text = "\n".join((page.extract_text() or "") for page in reader.pages[:3])
                except Exception:
                    text = ""
        except Exception:
            continue

        if not text:
            continue

        patterns = [
            rf"#\s*{re.escape(code[:4])}\s*{re.escape(code[4:])}\s*-\s*([^\n(]+)",
            rf"##\s*Course\s*Code\s*{re.escape(code)}.*?##\s*Course\s*Title\s*([^#\n]+)",
        ]
        title = ""
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
            if m:
                title = re.sub(r"\s+", " ", m.group(1)).strip()
                break

        if title:
            rel_under_knowledge = ""
            try:
                rel_under_knowledge = str(path.relative_to(Path(__file__).resolve().parent / "ECEknowledge")).replace("\\", "/")
            except Exception:
                rel_under_knowledge = _infer_source_relpath(str(path), {})

            encoded_relpath = quote(rel_under_knowledge, safe="/") if rel_under_knowledge else ""
            resource_url = f"/resource/ECEknowledge/{encoded_relpath}" if encoded_relpath else "#"
            direct_url = f"/ECEknowledge/{encoded_relpath}" if encoded_relpath else "#"

            return {
                "name": f"[{code}] official syllabus fact",
                "snippet": f"Official course title: {title}",
                "url": resource_url,
                "direct_url": direct_url,
                "source_relpath": rel_under_knowledge,
            }

    return None


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
    base_dir = Path(__file__).resolve().parent
    index_path = str(base_dir / _index_name_from_hub(hub_name))

    local_path = os.environ.get("EMBEDDING_MODEL_LOCAL_PATH", "").strip()
    if local_path:
        resolved = os.path.normpath(os.path.join(str(base_dir), local_path))
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
OFFICIAL_COURSE_CODES = _collect_official_course_codes()

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


def get_rag_context(query: str, k: int | None = None):
    """
    Retrieve relevant document chunks from the FAISS index for the given query.
    Returns a list of context dicts compatible with the server's streaming pipeline:
        [{ 'name': str, 'snippet': str, 'url': str }, ...]
    """
    if _vectorstore is None:
        return []

    query_course_code = _extract_course_code(query)
    query_course_code = _normalize_course_code(query_course_code) if query_course_code else ""
    query_code_not_official = bool(
        query_course_code and OFFICIAL_COURSE_CODES and query_course_code not in OFFICIAL_COURSE_CODES
    )

    try:
        final_k = max(1, int(k)) if k is not None else FAISS_FINAL_K
        candidate_k = max(FAISS_CANDIDATE_K, final_k)
        search_queries = [query]
        if query_course_code:
            search_queries.append(query_course_code)
            search_queries.append(f"{query_course_code} course code course title offering semester description")

        retrieved_map = {}
        per_query_k = max(final_k, candidate_k // max(1, len(search_queries)))
        # similarity_search_with_score returns (Document, score) tuples
        for search_query in search_queries:
            try:
                retrieved = _vectorstore.similarity_search_with_score(search_query, k=per_query_k)
            except Exception:
                continue
            for doc, raw_score in retrieved:
                key = (
                    str(doc.metadata.get("source", doc.metadata.get("file_path", ""))),
                    str(doc.metadata.get("chunk_id", "")),
                    doc.page_content[:240],
                )
                if key not in retrieved_map or raw_score < retrieved_map[key][1]:
                    retrieved_map[key] = (doc, raw_score)

        retrieved = list(retrieved_map.values())
    except Exception as e:
        logger.error(f"[FAISS RAG] Search failed: {e}")
        return []

    reranked = []
    for doc, raw_score in retrieved:
        metadata = doc.metadata or {}
        file_path = metadata.get("source", metadata.get("file_path", ""))
        doc_course_code = _normalize_course_code(str(metadata.get("course_code", "")))
        doc_department = str(metadata.get("department", "")).upper()
        doc_type = str(metadata.get("doc_type", "")).lower()
        source_name = str(metadata.get("source_name", "") or _safe_filename_from_path(file_path)).lower()
        source_quality = str(metadata.get("source_quality", "")).lower()
        course_conf = str(metadata.get("course_code_confidence", "")).lower()
        is_aggregated = bool(metadata.get("is_aggregated_syllabus")) or "aggregate" in source_quality
        if not is_aggregated and source_name.endswith(".docx") and "syllabus" in source_name:
            is_aggregated = True
        code_not_official = bool(doc_course_code and OFFICIAL_COURSE_CODES and doc_course_code not in OFFICIAL_COURSE_CODES)

        score = float(raw_score)
        text_window = f"{doc.page_content[:1200]}\n{file_path}"

        if is_aggregated:
            score += 0.18
        if course_conf in {"none", "ambiguous"}:
            score += 0.20
        if code_not_official and is_aggregated:
            score += 0.50

        if query_course_code:
            if doc_course_code and doc_course_code == query_course_code:
                score -= 0.45
            elif query_course_code in text_window.upper():
                score -= 0.25

            if doc_course_code and doc_course_code != query_course_code:
                score += 0.12

            if doc_department and query_course_code.startswith(doc_department):
                score -= 0.05

            # If the queried code is not present in official syllabus files,
            # avoid over-trusting reviews/secondary pages that can be outdated.
            if query_code_not_official:
                if doc_type == "course_review":
                    score += 0.80
                if doc_course_code == query_course_code and doc_type != "course_syllabus":
                    score += 0.50

        reranked.append((doc, raw_score, score))

    reranked.sort(key=lambda x: x[2])

    if query_code_not_official:
        filtered = []
        for doc, raw_score, rerank_score in reranked:
            metadata = doc.metadata or {}
            doc_type = str(metadata.get("doc_type", "")).lower()
            if doc_type == "course_review":
                continue
            file_path = str(metadata.get("source", metadata.get("file_path", "")))
            haystack = f"{doc.page_content[:3000]}\n{file_path}".upper()
            if query_course_code in haystack:
                filtered.append((doc, raw_score, rerank_score))

        if filtered:
            reranked = filtered
        else:
            return [{
                "name": f"{query_course_code} (verification note)",
                "snippet": (
                    f"No official course_syllabus file for {query_course_code} was found in the current "
                    "knowledge base. Mentions from secondary sources may be outdated; please verify with "
                    "the latest HKUST official course catalog."
                ),
                "url": "#",
                "direct_url": "#",
                "source_relpath": "",
            }]

    if query_course_code:
        exact_official = []
        fallback_official = []
        for doc, raw_score, rerank_score in reranked:
            metadata = doc.metadata or {}
            doc_type = str(metadata.get("doc_type", "")).lower()
            doc_course_code = _normalize_course_code(str(metadata.get("course_code", "")))
            if doc_type == "course_syllabus" and doc_course_code == query_course_code:
                exact_official.append((doc, raw_score, rerank_score))
            elif doc_type == "course_syllabus":
                fallback_official.append((doc, raw_score, rerank_score))

        if exact_official:
            reranked = exact_official
        elif fallback_official:
            reranked = fallback_official

    threshold = 1.2 if query_course_code else FAISS_SCORE_THRESHOLD

    # Debug
    print(f"[FAISS RAG Debug] Query: {query}")
    if query_course_code:
        print(f"[FAISS RAG Debug] Detected course code: {query_course_code}")
    for doc, raw_score, rerank_score in reranked[:final_k]:
        meta = doc.metadata or {}
        cc = meta.get("course_code", "")
        print(f"[FAISS RAG Debug] raw={raw_score:.4f}, rerank={rerank_score:.4f}, cc={cc} | {doc.page_content[:60]}...")

    context = []
    official_fact = _lookup_official_course_fact(query_course_code) if query_course_code else None
    if official_fact:
        context.append(official_fact)
    for doc, _, score in reranked[:final_k]:
        if score >= threshold:
            continue  # Too dissimilar — skip

        metadata = doc.metadata
        # LangChain PDF loaders store the source path in 'source'
        file_path = metadata.get("source", metadata.get("file_path", ""))

        source_relpath = _infer_source_relpath(file_path, metadata)
        encoded_relpath = quote(source_relpath, safe="/") if source_relpath else ""
        original_url = str(metadata.get("original_url", "")).strip()
        source_quality = str(metadata.get("source_quality", "")).lower()
        source_name = str(metadata.get("source_name", "") or _safe_filename_from_path(file_path)).lower()
        is_aggregated = bool(metadata.get("is_aggregated_syllabus")) or "aggregate" in source_quality
        if not is_aggregated and source_name.endswith(".docx") and "syllabus" in source_name:
            is_aggregated = True

        if "page" in metadata:
            # page is 0-indexed in LangChain loaders
            page_num = int(metadata["page"]) + 1
            name = f"Page {page_num}, {_safe_filename_from_path(file_path)}"
            url_suffix = f"#page={page_num}"
        else:
            name = _safe_filename_from_path(file_path)
            url_suffix = ""

        course_code = _normalize_course_code(str(metadata.get("course_code", "")))
        doc_type = metadata.get("doc_type", "")

        # Hard guardrail: hide aggregate-only codes that have no concrete course file support.
        if is_aggregated and course_code and OFFICIAL_COURSE_CODES and course_code not in OFFICIAL_COURSE_CODES:
            continue

        if query_code_not_official and course_code == query_course_code and str(doc_type).lower() == "course_review":
            continue

        if course_code:
            name = f"[{course_code}] {name}"
        if doc_type:
            name = f"{name} ({doc_type})"

        resource_url = f"/resource/ECEknowledge/{encoded_relpath}{url_suffix}" if encoded_relpath else "#"
        direct_url = f"/ECEknowledge/{encoded_relpath}{url_suffix}" if encoded_relpath else "#"
        citation_url = original_url if original_url else resource_url

        context.append({
            "name": name,
            "snippet": doc.page_content,
            "url": citation_url,
            "direct_url": direct_url,
            "source_relpath": source_relpath,
        })

    # De-duplicate by snippet content
    unique_context = list({entry["snippet"]: entry for entry in context}.values())

    return unique_context

