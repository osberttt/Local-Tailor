"""
Clean up all generated files, models, and user config.
Run before 'setup' for a fresh start.

Usage:
  python clean.py          # clean active shop (from config.SHOP)
  python clean.py --all    # clean all shops

Does NOT delete:
  shops/             — shop definitions (dimensions, examples, synthetic)
  venv/              — virtual environment
  localtailor/       — source code
  templates/         — report templates
"""

import argparse
import shutil
from pathlib import Path

from localtailor.config import SHOP

GENERATED_DATA_PATTERNS = [
    "comments_clean_*.json",
    "ground_truth_*.json",
    "predictions_*.json",
    "evaluation_*.json",
    "sensitivity_*.json",
    "embeddings_*.npy",
    "embedding_index_*.json",
]

PYCACHE = "__pycache__"


def clean_shop(shop: str) -> int:
    """Clean generated data and models for a specific shop. Returns count of removed items."""
    removed = 0

    # Generated data files in data/{shop}/
    data_dir = Path("data") / shop
    if data_dir.exists():
        for pattern in GENERATED_DATA_PATTERNS:
            for f in data_dir.glob(pattern):
                f.unlink()
                print(f"  Deleted {f}")
                removed += 1
        # Remove dir if empty
        if data_dir.exists() and not any(data_dir.iterdir()):
            data_dir.rmdir()
            print(f"  Removed empty {data_dir}/")

    # Trained models in models/{shop}/
    models_dir = Path("models") / shop
    if models_dir.exists():
        shutil.rmtree(models_dir)
        print(f"  Deleted {models_dir}/")
        removed += 1

    # Reports
    reports_dir = Path("reports")
    if reports_dir.exists():
        for f in reports_dir.glob(f"report_{shop}_*"):
            f.unlink()
            print(f"  Deleted {f}")
            removed += 1

    return removed


def clean(all_shops: bool = False):
    removed = 0

    if all_shops:
        # Clean all shop data directories
        for data_sub in Path("data").iterdir() if Path("data").exists() else []:
            if data_sub.is_dir():
                removed += clean_shop(data_sub.name)
        # Clean all model directories
        if Path("models").exists():
            shutil.rmtree(Path("models"))
            print("  Deleted models/")
            removed += 1
        # Clean reports
        if Path("reports").exists():
            shutil.rmtree(Path("reports"))
            print("  Deleted reports/")
            removed += 1
    else:
        print(f"  Cleaning shop: {SHOP}")
        removed += clean_shop(SHOP)

    # __pycache__ directories (always clean)
    for cache_dir in Path(".").rglob(PYCACHE):
        if "venv" not in cache_dir.parts:
            shutil.rmtree(cache_dir)
            print(f"  Deleted {cache_dir}/")
            removed += 1

    # Legacy flat files (from before multi-shop migration)
    legacy_patterns = [
        "data/comments_clean_*.json",
        "data/ground_truth_*.json",
        "data/predictions_*.json",
        "data/evaluation_*.json",
        "data/sensitivity_*.json",
        "data/embeddings_*.npy",
        "data/embedding_index_*.json",
        "data/examples.json",
    ]
    legacy_files = ["config/dimensions.yaml"]

    for pattern in legacy_patterns:
        for f in Path(".").glob(pattern):
            f.unlink()
            print(f"  Deleted legacy {f}")
            removed += 1
    for fp in legacy_files:
        p = Path(fp)
        if p.exists():
            p.unlink()
            print(f"  Deleted legacy {p}")
            removed += 1

    if removed == 0:
        print("  Nothing to clean.")
    else:
        print(f"\n  Cleaned {removed} items.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Tailor — Clean")
    parser.add_argument("--all", action="store_true", help="Clean all shops, not just active one")
    args = parser.parse_args()

    print("\n  LOCAL TAILOR — Clean\n")
    clean(all_shops=args.all)
    print()
