"""
localtailor/embedder.py
=======================
Encodes all comments into 384-dim sentence embeddings using all-MiniLM-L6-v2.
BERT is frozen permanently — runs once, cached to .npy.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np


def generate_embeddings(comments_path: Path, post_id: str, batch_size: int = 64) -> Path:
    """Encode all comments → embeddings_{post_id}.npy"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Run: pip install sentence-transformers")

    with open(comments_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    comments = data["comments"]
    texts = [c["message"] for c in comments]

    print(f"  Loading all-MiniLM-L6-v2 (~80MB, cached after first run)...")
    t0 = time.time()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    print(f"  Encoding {len(texts)} comments (batch_size={batch_size})...")
    t1 = time.time()
    embeddings = model.encode(
        texts, batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  Done in {time.time()-t1:.1f}s — shape: {embeddings.shape}")

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    emb_path = output_dir / f"embeddings_{post_id}.npy"
    idx_path = output_dir / f"embedding_index_{post_id}.json"

    np.save(emb_path, embeddings.astype(np.float32))
    index = [{"idx": i, "comment_id": c["id"]} for i, c in enumerate(comments)]
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"  Saved → {emb_path}")
    return emb_path


def load_embeddings(post_id: str) -> Tuple[np.ndarray, List[str]]:
    """Load saved embeddings. Returns (matrix, comment_ids)."""
    emb_path = Path(f"data/embeddings_{post_id}.npy")
    idx_path = Path(f"data/embedding_index_{post_id}.json")
    if not emb_path.exists():
        raise FileNotFoundError(f"Run generate_embeddings() first: {emb_path}")
    embeddings = np.load(emb_path)
    with open(idx_path) as f:
        index = json.load(f)
    return embeddings, [e["comment_id"] for e in index]
