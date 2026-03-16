"""
Clean up all generated files, models, and user config.
Run before 'demo' or 'setup' for a fresh start.

Usage:
  python clean.py

Does NOT delete:
  demo/              — dev reference config (read-only)
  venv/              — virtual environment
  localtailor/       — source code
  templates/         — report templates
"""

import shutil
from pathlib import Path

GENERATED_DATA = [
    "data/comments_clean_*.json",
    "data/ground_truth_*.json",
    "data/predictions_*.json",
    "data/evaluation_*.json",
    "data/sensitivity_*.json",
    "data/embeddings_*.npy",
    "data/embedding_index_*.json",
    "data/examples.json",
]

GENERATED_FILES = [
    "config/dimensions.yaml",
]

GENERATED_DIRS = [
    "models",
    "reports",
]

PYCACHE = "__pycache__"


def clean():
    removed = 0

    # Generated data files
    for pattern in GENERATED_DATA:
        for f in Path(".").glob(pattern):
            f.unlink()
            print(f"  Deleted {f}")
            removed += 1

    # Generated single files
    for fp in GENERATED_FILES:
        p = Path(fp)
        if p.exists():
            p.unlink()
            print(f"  Deleted {p}")
            removed += 1

    # Generated directories (models/, reports/)
    for d in GENERATED_DIRS:
        p = Path(d)
        if p.exists():
            shutil.rmtree(p)
            print(f"  Deleted {p}/")
            removed += 1

    # __pycache__ directories
    for cache_dir in Path(".").rglob(PYCACHE):
        if "venv" not in cache_dir.parts:
            shutil.rmtree(cache_dir)
            print(f"  Deleted {cache_dir}/")
            removed += 1

    if removed == 0:
        print("  Nothing to clean.")
    else:
        print(f"\n  Cleaned {removed} items.")


if __name__ == "__main__":
    print("\n  LOCAL TAILOR — Clean\n")
    clean()
    print()
