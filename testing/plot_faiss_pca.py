#!/usr/bin/env python3
"""
Visualize FAISS vector embeddings in 2D using PCA projection.
Vectors are colored by department/doc_type to identify clustering patterns.

Usage:
  python plot_faiss_pca.py                                    # default index
  python plot_faiss_pca.py --index faiss_index_bge-small-en-v1.5
  python plot_faiss_pca.py --n-vectors 1000                  # sample fewer vectors
"""

import pickle
import argparse
from pathlib import Path
from collections import Counter
import faiss
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Visualize FAISS vectors with PCA")
    parser.add_argument(
        "--index",
        default="faiss_index_all-MiniLM-L6-v2",
        help="Index folder name"
    )
    parser.add_argument(
        "--n-vectors",
        type=int,
        default=3000,
        help="Number of vectors to sample for visualization (default: 3000)"
    )
    parser.add_argument(
        "--color-by",
        choices=["department", "doc_type"],
        default="department",
        help="Color scheme: department or doc_type (default: department)"
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        print("ERROR: matplotlib and scikit-learn required")
        print("  Install with: pip install matplotlib scikit-learn")
        return

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
    print(f"FAISS PCA Visualization: {args.index}")
    print(f"{'='*70}\n")

    # Load FAISS index
    print(f"[1] Loading FAISS index...")
    idx = faiss.read_index(str(faiss_path))
    print(f"    Total vectors: {idx.ntotal}")

    # Load docstore
    print(f"[2] Loading metadata (docstore)...")
    with open(pkl_path, "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)
    print(f"    Docstore IDs: {len(index_to_docstore_id)}")

    # Reconstruct vectors
    n = min(args.n_vectors, idx.ntotal)
    print(f"[3] Reconstructing {n} vectors...")
    vecs = idx.reconstruct_n(0, n)
    print(f"    Shape: {vecs.shape}")

    # Extract labels
    print(f"[4] Extracting {args.color_by} labels...")
    labels = []
    for i in range(n):
        ds_id = index_to_docstore_id.get(i)
        label = "unknown"
        if ds_id is not None:
            doc = docstore.search(ds_id)
            md = getattr(doc, "metadata", {}) or {}
            if args.color_by == "department":
                label = md.get("department") or "unknown"
            else:  # doc_type
                label = md.get("doc_type") or "unknown"
        labels.append(label)

    # PCA
    print(f"[5] Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(vecs)
    print(f"    Explained variance: {pca.explained_variance_ratio_}")

    # Plot
    print(f"[6] Rendering plot...")
    top_labels = {k for k, _ in Counter(labels).most_common(8)}
    label_to_color = {}
    for l in set(labels):
        label_to_color[l] = l if l in top_labels else "other"

    plt.figure(figsize=(14, 9))
    for label in sorted(set(label_to_color.values())):
        mask = [label_to_color[x] == label for x in labels]
        plt.scatter(xy[mask, 0], xy[mask, 1], s=12, alpha=0.6, label=label)

    plt.title(f"FAISS Vector Embeddings (PCA) - {args.index}", fontsize=14, fontweight='bold')
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=11)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=11)
    plt.legend(markerscale=2, fontsize=10, title=args.color_by.capitalize())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

