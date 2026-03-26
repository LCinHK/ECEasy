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

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    TextLoader,
    BSHTMLLoader   # ← NEW: Added for HTML support
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ======== Configuration ========
DATA_PATH = Path("ECEknowledge")       # Source knowledge folder

# Files / patterns to skip (e.g. macOS metadata files)
SKIP_PATTERNS = {".DS_Store"}

# Image file extensions to catalog
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Metadata enrichment controls
PREPEND_METADATA_TO_CHUNK_TEXT = True

# Supports patterns like: COMP2011, COMP 2011, COMP-2011, COMP2011_Spring2025-26
COURSE_CODE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{4})\s*[-_]?\s*(\d{4}[A-Za-z]?)(?![A-Za-z0-9])")


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


def _detect_doc_type(rel_path: Path) -> str:
    rel = str(rel_path).replace("\\", "/").lower()
    if "course syllabus/" in rel:
        return "course_syllabus"
    if "program requirement/" in rel:
        return "program_requirement"
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

    return {
        "source_relpath": rel_posix,
        "source_name": file_path.name,
        "source_stem": file_path.stem,
        "doc_type": _detect_doc_type(rel_path),
        "department": dept,
        "course_code": course_code,
    }


def _index_name_from_hub(hub_name: str) -> str:
    short = hub_name.split("/")[-1]
    return f"faiss_index_{short}"


def _resolve_embedding_model() -> tuple[str, str]:
    hub_name = os.environ.get("EMBEDDING_MODEL_HUB_NAME", "all-MiniLM-L6-v2").strip()
    base_dir = Path(__file__).resolve().parent
    index_path = str(base_dir / _index_name_from_hub(hub_name))

    local_path = os.environ.get("EMBEDDING_MODEL_LOCAL_PATH", "").strip()
    if local_path:
        resolved = os.path.normpath(os.path.join(str(base_dir), local_path))
        if os.path.isdir(resolved):
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            print(f"[Embedding] Using local model folder: '{resolved}' (offline)")
            print(f"[Embedding] FAISS index will be saved to: '{index_path}'")
            return resolved, index_path

    print(f"[Embedding] Using HuggingFace Hub model: '{hub_name}' (requires internet on first run)")
    print(f"[Embedding] FAISS index will be saved to: '{index_path}'")
    return hub_name, index_path


def _extract_image_metadata(file_path: Path, data_path: Path) -> dict:
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

    doc_type = _detect_doc_type(rel_path)
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
    image_manifest = []
    image_files = sorted(data_path.rglob("*"))
    image_files = [f for f in image_files if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

    print(f"  Found {len(image_files)} image file(s)")
    for img_path in image_files:
        if img_path.name in SKIP_PATTERNS:
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
    pdf_files = sorted(data_path.rglob("*.pdf"))
    print(f"  Found {len(pdf_files)} PDF file(s)")
    for pdf_path in pdf_files:
        if pdf_path.name in SKIP_PATTERNS:
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
    docx_files = sorted(data_path.rglob("*.docx"))
    print(f"  Found {len(docx_files)} DOCX file(s)")
    for docx_path in docx_files:
        if docx_path.name in SKIP_PATTERNS:
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

    # --- TXT ---
    txt_files = sorted(data_path.rglob("*.txt"))
    print(f"  Found {len(txt_files)} TXT file(s)")
    for txt_path in txt_files:
        if txt_path.name in SKIP_PATTERNS:
            continue
        try:
            try:
                loader = TextLoader(str(txt_path), encoding="utf-8")
                docs = loader.load()
            except Exception:
                loader = TextLoader(str(txt_path), encoding="latin-1")
                docs = loader.load()
            structured_meta = _extract_structured_metadata(txt_path, data_path)
            for doc in docs:
                doc.metadata["source"] = str(txt_path)
                doc.metadata.update(structured_meta)
            all_docs.extend(docs)
            print(f"    [TXT]  {txt_path.relative_to(data_path)}  ({len(docs)} doc(s))")
        except Exception as e:
            skipped.append((str(txt_path), str(e)))
            print(f"    [TXT]  SKIP {txt_path.name}: {e}")

    # === NEW: HTML Support ===
    html_files = sorted(data_path.rglob("*.html")) + sorted(data_path.rglob("*.htm"))
    print(f"  Found {len(html_files)} HTML file(s)")
    for html_path in html_files:
        if html_path.name in SKIP_PATTERNS:
            continue
        try:
            # BSHTMLLoader extracts clean text from HTML (removes scripts, styles, etc.)
            loader = BSHTMLLoader(str(html_path))
            docs = loader.load()
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
            inferred = _extract_course_code(chunk.page_content[:1000])
            if inferred:
                chunk.metadata["course_code"] = _normalize_course_code(inferred)

        if PREPEND_METADATA_TO_CHUNK_TEXT:
            header_parts = []
            cc = chunk.metadata.get("course_code", "")
            dep = chunk.metadata.get("department", "")
            dt = chunk.metadata.get("doc_type", "")
            src = chunk.metadata.get("source_relpath", "")
            if cc:
                header_parts.append(f"COURSE_CODE={cc}")
            if dep:
                header_parts.append(f"DEPARTMENT={dep}")
            if dt:
                header_parts.append(f"DOC_TYPE={dt}")
            if src:
                header_parts.append(f"SOURCE={src}")

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

    # Remove old index for this model before saving
    old_index = Path(index_path)
    if old_index.exists():
        shutil.rmtree(old_index)
        print(f"Removed old index at '{index_path}'")

    vectorstore.save_local(index_path)
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