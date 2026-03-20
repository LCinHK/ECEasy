#!/usr/bin/env python3
"""
Debug FAISS retrieval quality and reranking behavior.
Tests sample queries (especially course-code queries) and shows retrieved chunks
with metadata and scores.

Usage:
  python query_debug.py                                       # use default queries
  python query_debug.py --queries "ELEC1100" "COMP2011"       # custom queries
  python query_debug.py --k 12                                # show top 12 results (default 8)
  python query_debug.py --index faiss_index_bge-small-en-v1.5
"""

import re
import argparse
from pathlib import Path
import sys
import numpy as np
def main():
    parser = argparse.ArgumentParser(description="Debug FAISS retrieval and reranking")
    parser.add_argument(
        "--index",
        default="faiss_index_all-MiniLM-L6-v2",
        help="Index folder name"
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=[
            "ELEC1100",
            "What is COMP2011? Why ELEC students need to take it"
        ],
        help="Test queries (space-separated)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Show top-k results (default: 8)"
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=40,
        help="Retrieve this many candidates before reranking (default: 40)"
    )
    args = parser.parse_args()

    # Setup paths
    project_root = Path("..")
    index_dir = project_root / args.index

    if not index_dir.exists():
        print(f"ERROR: Index directory not found: {index_dir}")
        return

    faiss_path = index_dir / "index.faiss"
    pkl_path = index_dir / "index.pkl"

    if not faiss_path.exists() or not pkl_path.exists():
        print(f"ERROR: Missing index files in {index_dir}")
        return

    # Import dependencies
    try:
        import pickle
        import faiss
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as e:
        print(f"ERROR: Missing dependencies: {e}")
        print("  pip install faiss-cpu langchain langchain-huggingface langchain-community")
        return

    print(f"\n{'='*80}")
    print(f"FAISS Retrieval Debug Tool")
    print(f"{'='*80}")
    print(f"Index:      {args.index}")
    print(f"Top-K:      {args.k}")
    print(f"Candidates: {args.candidate_k}")
    print(f"Queries:    {len(args.queries)}\n")

    # Load index + metadata
    print(f"[1] Loading index and metadata...")
    idx = faiss.read_index(str(faiss_path))
    with open(pkl_path, "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)

    # Load embeddings model (must match ingestion model!)
    model_name = args.index.replace("faiss_index_", "")
    print(f"[2] Loading embeddings model: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Helper functions (from faiss_rag.py)
    COURSE_CODE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{4})\s*[-_]?\s*(\d{4}[A-Za-z]?)(?![A-Za-z0-9])")

    def normalize_course_code(text: str) -> str:
        compact = re.sub(r"\s+", "", text).replace("-", "").replace("_", "").upper()
        m = COURSE_CODE_RE.search(compact)
        if not m:
            return ""
        return f"{m.group(1).upper()}{m.group(2).upper()}"

    def extract_course_code(text: str) -> str:
        m = COURSE_CODE_RE.search(text)
        if not m:
            return ""
        return f"{m.group(1).upper()}{m.group(2).upper()}"

    # Test each query
    for query_idx, query in enumerate(args.queries, 1):
        print(f"\n{'-'*80}")
        print(f"Query {query_idx}: {query}")
        print(f"{'-'*80}")

        query_code = extract_course_code(query)
        query_code_norm = normalize_course_code(query_code) if query_code else ""

        print(f"Detected course code: {query_code_norm if query_code_norm else '(none)'}")

        # Embed query
        query_vec = embeddings.embed_query(query)

        # Search
        print(f"\nSearching with k={args.candidate_k}...")
        query_vec_array = np.array([query_vec], dtype=np.float32)
        distances, indices = idx.search(query_vec_array, k=args.candidate_k)

        # For course-code queries, also retrieve with code-only text and merge candidates.
        merged = {}
        for idx_pos, raw_score in zip(indices[0], distances[0]):
            merged[int(idx_pos)] = float(raw_score)

        if query_code_norm:
            code_vec = embeddings.embed_query(query_code_norm)
            code_vec_array = np.array([code_vec], dtype=np.float32)
            d2, i2 = idx.search(code_vec_array, k=args.candidate_k)
            for idx_pos, raw_score in zip(i2[0], d2[0]):
                idx_pos = int(idx_pos)
                raw_score = float(raw_score)
                if idx_pos not in merged or raw_score < merged[idx_pos]:
                    merged[idx_pos] = raw_score

        # Rerank (same logic as faiss_rag.py)
        reranked = []
        for idx_pos, raw_score in merged.items():
            ds_id = index_to_docstore_id.get(int(idx_pos))
            if ds_id is None:
                continue

            doc = docstore.search(ds_id)
            md = getattr(doc, "metadata", {}) or {}
            doc_code = normalize_course_code(str(md.get("course_code", "")))
            doc_dept = str(md.get("department", "")).upper()
            source_relpath = str(md.get("source_relpath", ""))

            score = float(raw_score)
            text_window = f"{doc.page_content[:1200]}\n{md.get('source_relpath', '')}"

            # Apply boosts (from faiss_rag.py)
            if query_code_norm:
                if doc_code and doc_code == query_code_norm:
                    score -= 0.45
                elif query_code_norm in text_window.upper():
                    score -= 0.25

                if query_code_norm in source_relpath.upper():
                    score -= 0.35

                if doc_code and doc_code != query_code_norm:
                    score += 0.12

                if doc_dept and query_code_norm.startswith(doc_dept):
                    score -= 0.05

            reranked.append((int(idx_pos), raw_score, score, doc, md))

        reranked.sort(key=lambda x: x[2])

        # Show top-k
        print(f"\nTop {args.k} results (after reranking):")
        print(f"{'':<3} {'Raw Score':<12} {'Reranked':<12} {'Code':<12} {'Type':<18} {'Match':<6} Snippet")
        print(f"{'='*140}")

        for rank, (idx_pos, raw_score, rerank_score, doc, md) in enumerate(reranked[:args.k], 1):
            code = md.get("course_code", "")
            dtype = md.get("doc_type", "general")[:16]
            source = md.get("source_relpath", "")

            # Check if code matches
            match = "✓ YES" if query_code_norm and code and code.upper() == query_code_norm else ""

            snippet = doc.page_content[:80].replace("\n", " ")

            print(
                f"{rank:<3} {raw_score:<12.4f} {rerank_score:<12.4f} {code:<12} {dtype:<18} {match:<6} {snippet}..."
            )
            print(f"     └─ Source: {source}")
            print(f"     └─ Dept: {md.get('department', 'N/A'):<6} DocType: {dtype:<16} ChunkID: {md.get('chunk_id', 'N/A')}")

        print()

    print(f"\n{'='*80}")
    print(f"Debug complete.\n")

if __name__ == "__main__":
    main()

