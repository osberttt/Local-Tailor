"""
localtailor/synthetic.py
========================
Dispatches to the active shop's synthetic dataset generator.

Each shop defines its own comments and ground truth in:
  shops/{SHOP}/synthetic.py  →  COMMENTS_RAW, ALL_DIMS

This module loads the shop-specific data and generates the standard
comments_clean + ground_truth JSON files into data/{SHOP}/.
"""

from __future__ import annotations
import importlib
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict

from localtailor.config import SHOP, shop_paths


def generate_synthetic_dataset(shop: str | None = None, output_dir: str | None = None) -> tuple[Path, Path]:
    """Generate synthetic dataset and ground truth files for a shop.

    Args:
        shop:       Shop name (defaults to config.SHOP).
        output_dir: Override output directory (defaults to data/{shop}/).

    Returns:
        (comments_path, ground_truth_path)
    """
    s = shop or SHOP
    paths = shop_paths(s)
    out_dir = output_dir or paths["data_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Import the shop's synthetic module
    mod = importlib.import_module(f"shops.{s}.synthetic")
    comments_raw = mod.COMMENTS_RAW
    all_dims = mod.ALL_DIMS

    comments = []
    ground_truth = []

    for i, (message, gt) in enumerate(comments_raw):
        comment_id = f"c{i+1:03d}"
        comments.append({
            "id": comment_id,
            "idx": i,
            "message": message,
            "created_time": "2025-01-15T00:00:00",
            "like_count": 0,
            "comment_count": 0,
        })
        ground_truth.append({
            "comment_id": comment_id,
            "message": message,
            "ground_truth": gt,
        })

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_comments": len(comments),
        "dimensions": all_dims,
        "description": f"Synthetic {s} shop comment dataset for Local Tailor demo",
        "shop": s,
    }

    comments_data = {"metadata": metadata, "comments": comments}
    gt_data = {"metadata": metadata, "ground_truth": ground_truth}

    comments_path = Path(out_dir) / "comments_clean_demo.json"
    gt_path = Path(out_dir) / "ground_truth_demo.json"

    with open(comments_path, "w", encoding="utf-8") as f:
        json.dump(comments_data, f, ensure_ascii=False, indent=2)

    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, ensure_ascii=False, indent=2)

    # Print distribution summary
    print(f"\n  -- Synthetic Dataset Summary ({s} shop) ------------------")
    print(f"  Total comments: {len(comments)}")
    for dim in all_dims:
        value_counts: Dict = {}
        for item in ground_truth:
            v = item["ground_truth"][dim]["value"]
            value_counts[v] = value_counts.get(v, 0) + 1
        non_na = {k: v for k, v in value_counts.items() if k != "N/A"}
        na_count = value_counts.get("N/A", 0)
        print(f"  {dim:15s}: {sum(non_na.values())} classified, {na_count} N/A")
    print(f"  ---------------------------------------------------------\n")

    return comments_path, gt_path


if __name__ == "__main__":
    generate_synthetic_dataset()
    print("Done.")
