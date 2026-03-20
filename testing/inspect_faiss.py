#!/usr/bin/env python3
"""
Inspect FAISS index statistics and metadata distribution.

Usage:
  python inspect_faiss.py                                    # default index
  python inspect_faiss.py --index faiss_index_bge-small-en-v1.5
"""

import pickle
import argparse
from pathlib import Path
from collections import Counter
import faiss
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Inspect FAISS index stats and metadata")
    parser.add_argument(
        "--index",
        default="faiss_index_all-MiniLM-L6-v2",
        help="Index folder name (default: faiss_index_all-MiniLM-L6-v2)"
    )
    args = parser.parse_args()

    index_dir = Path("..") / args.index
    if not index_dir.exists():
        print(f"ERROR: Index directory not found: {index_dir}")
        return

    faiss_path = index_dir / "index.faiss"
    pkl_path = index_dir / "index.pkl"

    if not faiss_path.exists() or not pkl_path.exists():
        print(f"ERROR: Missing index files in {index_dir}")
        return

    print(f"\n{'='*70}")
    print(f"FAISS Index Inspector: {args.index}")
    print(f"{'='*70}\n")

    # ======== Read FAISS Index ========
    print("[1] FAISS Vector Index")
    print(f"    Path: {faiss_path}")
    idx = faiss.read_index(str(faiss_path))
    print(f"    Total vectors (ntotal): {idx.ntotal}")
    print(f"    Vector dimension: {idx.d}")
    print(f"    Index type: {type(idx).__name__}")
    metric = getattr(idx, "metric_type", None)
    if metric:
        print(f"    Metric type: {metric}")

    # Sample reconstructed vectors
    n_sample = min(2000, idx.ntotal)
    if n_sample > 0 and hasattr(idx, "reconstruct_n"):
        vecs = idx.reconstruct_n(0, n_sample)
        norms = np.linalg.norm(vecs, axis=1)
        print(f"\n    Sample vector norms ({n_sample} vectors):")
        print(f"      mean: {norms.mean():.6f}")
        print(f"      std:  {norms.std():.6f}")
        print(f"      min:  {norms.min():.6f}")
        print(f"      max:  {norms.max():.6f}")

    # ======== Read Metadata (PKL) ========
    print(f"\n[2] LangChain Docstore (index.pkl)")
    print(f"    Path: {pkl_path}")
    with open(pkl_path, "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)

    print(f"    Docstore IDs: {len(index_to_docstore_id)}")

    # ======== Metadata Statistics ========
    print(f"\n[3] Metadata Distribution")

    doc_type_counter = Counter()
    dept_counter = Counter()
    course_counter = Counter()
    source_counter = Counter()

    n_peek = min(10000, len(index_to_docstore_id))
    for _i in range(n_peek):
        ds_id = index_to_docstore_id.get(_i)
        if ds_id is None:
            continue
        doc = docstore.search(ds_id)
        md = getattr(doc, "metadata", {}) or {}

        doc_type_counter[md.get("doc_type", "unknown")] += 1
        dept_counter[md.get("department", "unknown")] += 1

        cc = md.get("course_code", "")
        if cc:
            course_counter[cc] += 1

        src = md.get("source_relpath", "unknown")
        if src:
            source_counter[src] += 1

    print(f"\n    [a] Document Types (top 15):")
    for k, v in doc_type_counter.most_common(15):
        pct = 100.0 * v / n_peek
        print(f"        {k:30s}: {v:6d} ({pct:5.1f}%)")

    print(f"\n    [b] Departments (top 15):")
    for k, v in dept_counter.most_common(15):
        pct = 100.0 * v / n_peek
        print(f"        {k:10s}: {v:6d} ({pct:5.1f}%)")

    print(f"\n    [c] Course Codes (top 30):")
    for k, v in course_counter.most_common(30):
        print(f"        {k:12s}: {v:6d}")

    if not course_counter:
        print(f"        (No course codes found)")

    print(f"\n    [d] Source Files (top 20):")
    for k, v in source_counter.most_common(20):
        pct = 100.0 * v / n_peek
        print(f"        {v:6d} ({pct:5.1f}%) {k}")

    print(f"\n{'='*70}")
    print(f"Summary: {idx.ntotal} vectors, {len(index_to_docstore_id)} docstore entries")
    print(f"         {len(dept_counter)} departments, {len(course_counter)} course codes")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

