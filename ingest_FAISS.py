"""
Ingestion script for ECEasy FAISS knowledge base.
Reads all .pdf, .docx, .txt, .html, and image files from ECEknowledge/ and builds
(or rebuilds) the FAISS index at faiss_index_MODELNAME.

Also generates an image manifest (JSON) for quick frontend/backend reference.

Dependencies (install before running):
    pip install pypdf docx2txt faiss-cpu langchain-community langchain-huggingface 
               langchain-text-splitters pillow beautifulsoup4
"""

import os
import re
import shutil
import json
from pathlib import Path
from tempfile import mkdtemp

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from bs4 import BeautifulSoup

# ======== Configuration ========
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "ECEknowledge"  # Source knowledge folder

# Files / patterns to skip (e.g. macOS metadata files)
SKIP_PATTERNS = {".DS_Store", "Thumbs.db"}

# Image file extensions to catalog
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# Metadata enrichment controls
PREPEND_METADATA_TO_CHUNK_TEXT = True
TEXT_EXTENSIONS = {".txt", ".md", ".csv"}

# Supports patterns like: COMP2011, COMP 2011, COMP-2011, COMP2011_Spring2025-26
COURSE_CODE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{4})\s*[-_]?\s*(\d{4}[A-Za-z]?)(?![A-Za-z0-9])")
URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
COURSE_CODE_LINE_RE = re.compile(r"(?:^|\n)\s*##\s*Course\s*Code\s*\n+\s*([A-Za-z]{4}\s*[-_]?\s*\d{4}[A-Za-z]?)", re.IGNORECASE)
COURSE_HEADER_RE = re.compile(r"(?:^|\n)\s*#\s*([A-Za-z]{4}\s*[-_]?\s*\d{4}[A-Za-z]?)\b")


def _normalize_course_code(raw: str) -> str:
    compact = re.sub(r"\s+", "", raw).replace("-", "").replace("_", "").upper()
    m = COURSE_CODE_RE.search(compact)
    if not m:
        return ""
    return f"{m.group(1).upper()}{m.group(2).upper()}"


def _extract_course_code(text: str) -> str:
    m = COURSE_CODE_RE.search(text)
    if not m:
        return ""
    return f"{m.group(1).upper()}{m.group(2).upper()}"


def _extract_all_course_codes(text: str) -> list[str]:
    codes = []
    seen = set()
    for m in COURSE_CODE_RE.finditer(text or ""):
        code = f"{m.group(1).upper()}{m.group(2).upper()}"
        if code in seen:
            continue
        seen.add(code)
        codes.append(code)
    return codes


def _extract_confident_course_code(chunk_text: str) -> str:
    """
    Extract a course code only when it appears in explicit course-identity markers.
    This avoids assigning wrong codes from prerequisite/exclusion lines.
    """
    text = chunk_text or ""

    labeled = [
        _normalize_course_code(m.group(1))
        for m in COURSE_CODE_LINE_RE.finditer(text)
        if _normalize_course_code(m.group(1))
    ]
    if len(set(labeled)) == 1:
        return labeled[0]

    headers = [
        _normalize_course_code(m.group(1))
        for m in COURSE_HEADER_RE.finditer(text[:1200])
        if _normalize_course_code(m.group(1))
    ]
    if len(set(headers)) == 1:
        return headers[0]

    return ""


def _should_skip_file(file_path: Path) -> bool:
    name = file_path.name
    return (
        name in SKIP_PATTERNS
        or name.startswith("~$")
        or name.startswith(".")
    )


def _normalized_relpath_for_dedupe(file_path: Path, data_path: Path) -> str:
    rel = str(file_path.relative_to(data_path)).replace("\\", "/").lower()
    # Collapse common duplicate suffixes like " (1)", " (2)" before extension.
    rel = re.sub(r"\s*\(\d+\)(?=\.[a-z0-9]+$)", "", rel)
    return rel


def _deduplicate_files(file_paths: list[Path], data_path: Path) -> list[Path]:
    seen = set()
    deduped = []
    for path in sorted(file_paths):
        key = _normalized_relpath_for_dedupe(path, data_path)
        if key in seen:
            print(f"    [SKIP] duplicate candidate: {path.relative_to(data_path)}")
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _detect_doc_type(rel_path: Path) -> str:
    rel = str(rel_path).replace("\\", "/").lower()
    if rel.startswith("ustspace_reviews/"):
        return "course_review"
    if rel.startswith("ust_ranking/"):
        return "university_ranking"
    if rel.startswith("course_syllabus/") or "course syllabus/" in rel:
        return "course_syllabus"
    if rel.startswith("program_requirements/") or "program requirement/" in rel:
        return "program_requirement"
    if rel.startswith("academic_policies/"):
        return "academic_policy"
    if rel.startswith("student_guides/"):
        return "student_guide"
    if rel.startswith("ece_study_plans/"):
        return "study_plan"
    if rel.startswith("fyp_t_coop/"):
        return "fyp_t_coop"
    if rel.startswith("course_materials"):
        return "course_material"
    if "faq" in rel:
        return "faq"
    if "common core" in rel:
        return "common_core"
    if "requirements" in rel or "requirement" in rel:
        return "requirement"
    return "general"


def _extract_structured_metadata(file_path: Path, data_path: Path) -> dict:
    rel_path = file_path.relative_to(data_path)
    rel_posix = str(rel_path).replace("\\", "/")

    course_code = _extract_course_code(file_path.stem)
    if not course_code:
        course_code = _extract_course_code(rel_posix)

    dept = ""
    if course_code:
        dept = course_code[:4]
    else:
        for part in rel_path.parts:
            up = part.upper()
            if re.fullmatch(r"[A-Z]{4}", up):
                dept = up
                break

    # Handle files like MATH/2011.pdf where course prefix is implied by folder.
    if not course_code and dept:
        stem_match = re.fullmatch(r"(\d{4}[A-Za-z]?)", file_path.stem.strip())
        if stem_match:
            course_code = f"{dept}{stem_match.group(1).upper()}"

    section = rel_path.parts[0].lower() if rel_path.parts else ""
    subsection = rel_path.parts[1].lower() if len(rel_path.parts) > 1 else ""

    source_origin = ""
    if section == "ustspace_reviews":
        source_origin = "ustspace"
    elif section == "ust_ranking":
        source_origin = "ust_rankings"

    stem_lower = file_path.stem.lower()
    is_aggregated_syllabus = (
        file_path.suffix.lower() == ".docx"
        and "syllabus" in stem_lower
        and not course_code
    )
    source_quality = "aggregate_low_trust" if is_aggregated_syllabus else "high"

    course_code_confidence = "high" if course_code else "none"

    return {
        "source_relpath": rel_posix,
        "source_name": file_path.name,
        "source_stem": file_path.stem,
        "knowledge_section": section,
        "knowledge_subsection": subsection,
        "source_origin": source_origin,
        "doc_type": _detect_doc_type(rel_path),
        "department": dept,
        "course_code": course_code,
        "course_code_confidence": course_code_confidence,
        "is_aggregated_syllabus": is_aggregated_syllabus,
        "source_quality": source_quality,
    }


def _load_html_as_document(html_path: Path) -> list[Document]:
    """Parse HTML to clean text while dropping scripts/styles noise from saved web pages."""
    raw = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")

    original_url = ""
    # Prefer explicit URL in HTML comments, commonly used in saved pages as provenance.
    comment_blocks = re.findall(r"<!--(.*?)-->", raw, flags=re.DOTALL)
    for block in comment_blocks:
        m = URL_RE.search(block)
        if m:
            original_url = m.group(0).strip()
            break
    if not original_url:
        canonical = soup.find("link", rel=lambda x: x and "canonical" in str(x).lower())
        if canonical and canonical.get("href"):
            original_url = str(canonical.get("href")).strip()
    if not original_url:
        meta_og = soup.find("meta", attrs={"property": "og:url"})
        if meta_og and meta_og.get("content"):
            original_url = str(meta_og.get("content")).strip()

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    content_root = soup.find("main") or soup.find("article") or soup.body or soup
    title = (soup.title.string or "").strip() if soup.title else ""
    text = "\n".join(content_root.stripped_strings)

    cleaned_lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if len(line) > 220 and "{" in line and "}" in line:
            continue
        cleaned_lines.append(line)

    if title:
        cleaned_lines.insert(0, f"Title: {title}")

    cleaned_text = "\n".join(cleaned_lines)
    metadata = {}
    if title:
        metadata["title"] = title
    if original_url:
        metadata["original_url"] = original_url
    return [Document(page_content=cleaned_text, metadata=metadata)]


def _load_plain_text_file(path: Path) -> list[Document]:
    try:
        return TextLoader(str(path), encoding="utf-8").load()
    except Exception:
        return TextLoader(str(path), encoding="latin-1").load()


def _index_name_from_hub(hub_name: str) -> str:
    """
    Derives a filesystem-safe FAISS index folder name from a Hub model name.
      "BAAI/bge-small-en-v1.5"  →  "faiss_index_bge-small-en-v1.5"
      "all-MiniLM-L6-v2"        →  "faiss_index_all-MiniLM-L6-v2"
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
    each other. Must stay in sync with faiss_rag.py.
    """
    hub_name = os.environ.get("EMBEDDING_MODEL_HUB_NAME", "all-MiniLM-L6-v2").strip()
    index_path = str(BASE_DIR / _index_name_from_hub(hub_name))

    local_path = os.environ.get("EMBEDDING_MODEL_LOCAL_PATH", "").strip()
    if local_path:
        resolved = os.path.normpath(os.path.join(str(BASE_DIR), local_path))
        if os.path.isdir(resolved):
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            print(f"[Embedding] Using local model folder: '{resolved}' (offline)")
            print(f"[Embedding] FAISS index will be saved to: '{index_path}'")
            return resolved, index_path
        else:
            print(f"[Embedding] WARNING: EMBEDDING_MODEL_LOCAL_PATH '{local_path}' "
                  f"(resolved: '{resolved}') not found — falling back to HuggingFace Hub.")

    print(f"[Embedding] Using HuggingFace Hub model: '{hub_name}' (requires internet on first run)")
    print(f"[Embedding] FAISS index will be saved to: '{index_path}'")
    return hub_name, index_path


def _extract_image_metadata(file_path: Path, data_path: Path) -> dict:
    """
    Extract metadata from image file name and path.
    Example: "./ECEknowledge/course syllabus/common core courese/Common_Core_Course.png"
    → { "source_relpath": "course syllabus/common core courese/Common_Core_Course.png",
        "source_name": "Common_Core_Course.png",
        "doc_type": "course_requirement",
        "department": "common_core",
        "course_code": "",
        "description_from_filename": "Common Core Course" }
    """
    rel_path = file_path.relative_to(data_path)
    rel_posix = str(rel_path).replace("\\", "/")
    # Extract course code if present in filename or path
    course_code = _extract_course_code(file_path.stem)
    if not course_code:
        course_code = _extract_course_code(rel_posix)

    dept = ""
    if course_code:
        dept = course_code[:4]
    else:
        for part in rel_path.parts:
            up = part.upper()
            if re.fullmatch(r"[A-Z]{4}", up):
                dept = up
                break
    # Infer doc type from path
    doc_type = _detect_doc_type(rel_path)
    # Extract human-readable description from filename (e.g., "Common_Core_Course.png" → "Common Core Course")
    stem_clean = file_path.stem.replace("_", " ").replace("-", " ")

    return {
        "source_relpath": rel_posix,
        "source_name": file_path.name,
        "source_stem": file_path.stem,
        "file_size_bytes": file_path.stat().st_size,
        "doc_type": doc_type,
        "department": dept,
        "course_code": course_code,
        "description": stem_clean,
    }


def load_all_images(data_path: Path) -> list[dict]:
    """
    Scan for all image files (.png, .jpg, .jpeg) in data_path recursively.
    Return list of image metadata dicts (not embedded, just cataloged).
    """
    image_manifest = []
    image_files = sorted(data_path.rglob("*"))
    image_files = [f for f in image_files if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
    image_files = _deduplicate_files(image_files, data_path)

    print(f"  Found {len(image_files)} image file(s)")
    for img_path in image_files:
        if _should_skip_file(img_path):
            continue
        try:
            metadata = _extract_image_metadata(img_path, data_path)
            image_manifest.append(metadata)
            print(f"    [IMG]  {metadata['source_relpath']}  ({metadata.get('file_size_bytes', 0) / 1024:.1f} KB)")
        except Exception as e:
            print(f"    [IMG]  SKIP {img_path.name}: {e}")

    return image_manifest


def load_all_documents(data_path: Path):
    """
    Load all .pdf, .docx, .txt, and .html files recursively from data_path.
    """
    all_docs = []
    skipped = []

    # --- PDFs ---
    pdf_files = _deduplicate_files(list(data_path.rglob("*.pdf")), data_path)
    print(f"  Found {len(pdf_files)} PDF file(s)")
    for pdf_path in pdf_files:
        if _should_skip_file(pdf_path):
            continue
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            structured_meta = _extract_structured_metadata(pdf_path, data_path)
            for doc in docs:
                doc.metadata["source"] = str(pdf_path)
                doc.metadata.update(structured_meta)
            all_docs.extend(docs)
            print(f"    [PDF]  {pdf_path.relative_to(data_path)}  ({len(docs)} pages)")
        except Exception as e:
            skipped.append((str(pdf_path), str(e)))
            print(f"    [PDF]  SKIP {pdf_path.name}: {e}")

    # --- DOCX ---
    docx_files = _deduplicate_files(list(data_path.rglob("*.docx")), data_path)
    print(f"  Found {len(docx_files)} DOCX file(s)")
    for docx_path in docx_files:
        if _should_skip_file(docx_path):
            continue
        try:
            loader = Docx2txtLoader(str(docx_path))
            docs = loader.load()
            structured_meta = _extract_structured_metadata(docx_path, data_path)
            for doc in docs:
                doc.metadata["source"] = str(docx_path)
                doc.metadata.update(structured_meta)
            all_docs.extend(docs)
            print(f"    [DOCX] {docx_path.relative_to(data_path)}  ({len(docs)} doc(s))")
        except Exception as e:
            skipped.append((str(docx_path), str(e)))
            print(f"    [DOCX] SKIP {docx_path.name}: {e}")

    # --- Plain text-like files (TXT/MD/CSV) ---
    text_files = _deduplicate_files(
        [p for p in data_path.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS],
        data_path,
    )
    print(f"  Found {len(text_files)} text file(s) ({', '.join(sorted(TEXT_EXTENSIONS))})")
    for text_path in text_files:
        if _should_skip_file(text_path):
            continue
        try:
            docs = _load_plain_text_file(text_path)
            structured_meta = _extract_structured_metadata(text_path, data_path)
            for doc in docs:
                doc.metadata["source"] = str(text_path)
                doc.metadata.update(structured_meta)
            all_docs.extend(docs)
            print(f"    [TXT]  {text_path.relative_to(data_path)}  ({len(docs)} doc(s))")
        except Exception as e:
            skipped.append((str(text_path), str(e)))
            print(f"    [TXT]  SKIP {text_path.name}: {e}")

    # --- HTML ---
    html_files = _deduplicate_files(
        list(data_path.rglob("*.html")) + list(data_path.rglob("*.htm")),
        data_path,
    )
    print(f"  Found {len(html_files)} HTML file(s)")
    for html_path in html_files:
        if _should_skip_file(html_path):
            continue
        try:
            docs = _load_html_as_document(html_path)
            structured_meta = _extract_structured_metadata(html_path, data_path)
            for doc in docs:
                doc.metadata["source"] = str(html_path)
                doc.metadata.update(structured_meta)
            all_docs.extend(docs)
            print(f"    [HTML] {html_path.relative_to(data_path)}  ({len(docs)} doc(s))")
        except Exception as e:
            skipped.append((str(html_path), str(e)))
            print(f"    [HTML] SKIP {html_path.name}: {e}")

    if skipped:
        print(f"\n  Warning: {len(skipped)} file(s) could not be loaded:")
        for path, err in skipped:
            print(f"    - {path}: {err}")

    return all_docs


def main():
    if not DATA_PATH.exists() or not any(DATA_PATH.iterdir()):
        print(f"Error: Folder '{DATA_PATH}' is empty or doesn't exist.")
        print("→ Create it and add at least one .txt / .docx / .pdf / .html file")
        return

    print(f"Loading documents from '{DATA_PATH}'...")
    docs = load_all_documents(DATA_PATH)
    print(f"\nTotal raw pages/docs loaded: {len(docs)}")

    # Load images
    print(f"\nCataloging images from '{DATA_PATH}'...")
    image_manifest = load_all_images(DATA_PATH)
    print(f"Total images cataloged: {len(image_manifest)}")

    if len(docs) == 0:
        print("No documents loaded → nothing to index. Add files and retry.")
        return

    # Split into chunks
    print("\nSplitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)

    # Enrich every chunk with stable IDs and searchable metadata text.
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        chunk.metadata["chunk_total"] = len(chunks)

        if not chunk.metadata.get("course_code"):
            inferred = _extract_confident_course_code(chunk.page_content[:1500])
            if inferred:
                chunk.metadata["course_code"] = inferred
                chunk.metadata["course_code_confidence"] = "inferred_heading"

        # Guardrail: if chunk mentions multiple different course codes, avoid overconfident tagging.
        all_codes = _extract_all_course_codes(chunk.page_content[:2000])
        current_code = _normalize_course_code(str(chunk.metadata.get("course_code", "")))
        if current_code and len(all_codes) >= 2 and current_code not in all_codes[:1]:
            chunk.metadata["course_code"] = ""
            chunk.metadata["course_code_confidence"] = "ambiguous"

        if PREPEND_METADATA_TO_CHUNK_TEXT:
            header_parts = []
            cc = chunk.metadata.get("course_code", "")
            dep = chunk.metadata.get("department", "")
            dt = chunk.metadata.get("doc_type", "")
            src = chunk.metadata.get("source_relpath", "")
            sec = chunk.metadata.get("knowledge_section", "")
            sub = chunk.metadata.get("knowledge_subsection", "")
            origin = chunk.metadata.get("source_origin", "")
            quality = chunk.metadata.get("source_quality", "")
            cc_conf = chunk.metadata.get("course_code_confidence", "")
            if cc:
                header_parts.append(f"COURSE_CODE={cc}")
            if dep:
                header_parts.append(f"DEPARTMENT={dep}")
            if dt:
                header_parts.append(f"DOC_TYPE={dt}")
            if origin:
                header_parts.append(f"ORIGIN={origin}")
            if quality:
                header_parts.append(f"SOURCE_QUALITY={quality}")
            if cc_conf:
                header_parts.append(f"COURSE_CODE_CONFIDENCE={cc_conf}")
            if src:
                header_parts.append(f"SOURCE={src}")
            if sec:
                header_parts.append(f"SECTION={sec}")
            if sub:
                header_parts.append(f"SUBSECTION={sub}")

            if header_parts:
                chunk.page_content = f"[META] {' | '.join(header_parts)}\n\n{chunk.page_content}"

    print(f"Created {len(chunks)} chunks")

    # Resolve embedding model and derived index path
    model_name, index_path = _resolve_embedding_model()

    # Embed and store in FAISS
    print("\nLoading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Embedding and building FAISS index (this may take a while)...")
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    # Save to temp directory first, then atomically replace old index.
    temp_parent = Path(mkdtemp(prefix="eceasy_faiss_build_", dir=str(BASE_DIR)))
    temp_index_path = temp_parent / Path(index_path).name
    vectorstore.save_local(str(temp_index_path))

    old_index = Path(index_path)
    backup_index = old_index.with_name(f"{old_index.name}_backup")
    if backup_index.exists():
        shutil.rmtree(backup_index, ignore_errors=True)

    if old_index.exists():
        old_index.rename(backup_index)

    temp_index_path.rename(old_index)

    if backup_index.exists():
        shutil.rmtree(backup_index, ignore_errors=True)
    shutil.rmtree(temp_parent, ignore_errors=True)
    print(f"\nDone! FAISS index saved to: '{index_path}' ({len(chunks)} vectors)")

    # Save image manifest
    if image_manifest:
        manifest_path = Path(index_path) / "image_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(image_manifest, f, indent=2, ensure_ascii=False)
        print(f"Image manifest saved to: '{manifest_path}' ({len(image_manifest)} images)")
    else:
        print("No images found to catalog.")


if __name__ == "__main__":
    main()