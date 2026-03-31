"""
Local Tailor — Pipeline Entry Point
=====================================
Usage:
  python run_pipeline.py user         # download models + launch empty UI for user to configure
  python run_pipeline.py setup        # generate synthetic data + train + predict + eval + UI
  python run_pipeline.py retrain      # retrain models after editing dimensions/examples + predict + UI
  python run_pipeline.py predict      # load existing models + predict on current comments + UI
  python run_pipeline.py load-data    # just launch the Streamlit UI

Shop switching:
  Set SHOP in localtailor/config.py (e.g. SHOP = "pillow" or SHOP = "shoe").
  All data, models, and config paths resolve automatically from shops/{SHOP}/.

Files:
  shops/{SHOP}/dimensions.yaml  — dimension config for the active shop
  shops/{SHOP}/examples.json    — training examples for the active shop
  shops/{SHOP}/synthetic.py     — synthetic comments + ground truth definitions
  data/{SHOP}/                  — generated data (predictions, evaluation, etc.)
  models/{SHOP}/                — trained SetFit models
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from localtailor.config import SHOP, shop_paths

MODES = ["user", "setup", "retrain", "predict", "load-data"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Local Tailor Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Modes:",
            "  user         Download models, launch empty UI for initial configuration",
            "  setup        Generate synthetic data, train, predict, evaluate, launch UI",
            "  retrain      Retrain all models after editing dimensions/examples, then predict + UI",
            "  predict      Load existing models, run predictions on current comments, launch UI",
            "  load-data    Skip all processing, just launch the Streamlit dashboard",
        ]),
    )
    p.add_argument("mode", choices=MODES, help="Pipeline mode")
    return p.parse_args()


def launch_ui():
    print("\n" + "=" * 60)
    print("  Launching Streamlit UI...")
    print("  Open: http://localhost:8501")
    print("=" * 60 + "\n")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "localtailor/app.py",
            "--server.headless", "false",
        ])
    except KeyboardInterrupt:
        print("\n  Streamlit stopped.")


def _model_cached(model_name: str) -> bool:
    """Check if a HuggingFace model is already cached locally."""
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(model_name, "config.json")
        return isinstance(result, str)
    except Exception:
        return False


def cleanup_models(models_dir: str):
    """Delete all trained models for the active shop."""
    p = Path(models_dir)
    if p.exists():
        shutil.rmtree(p)
        print(f"  Cleaned: {p}")
    p.mkdir(parents=True, exist_ok=True)


def download_models():
    """Pre-download HuggingFace models if not already cached."""

    models = [
        ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2 (~80MB)"),
        ("deepset/roberta-base-squad2", "roberta-base-squad2 (~500MB)"),
    ]

    all_cached = True
    for model_id, label in models:
        if _model_cached(model_id):
            print(f"      {label} — already cached")
        else:
            all_cached = False
            print(f"      Downloading {label}...")
            if "sentence-transformers" in model_id:
                from sentence_transformers import SentenceTransformer
                SentenceTransformer(model_id)
            else:
                from transformers import pipeline as hf_pipeline
                hf_pipeline("question-answering", model=model_id, device=-1)
            print(f"      Done.")

    if all_cached:
        print("  All models already cached.\n")
    else:
        print("\n  All models downloaded and cached.\n")


def main():
    args = parse_args()
    mode = args.mode
    paths = shop_paths()

    # Ensure output dirs
    Path(paths["data_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["models_dir"]).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  LOCAL TAILOR")
    print(f"  Shop: {SHOP}")
    print("  Comment intelligence fitted to your business.")
    print("=" * 60 + "\n")

    # ── user: download models + launch empty UI ──────────────────────────────
    if mode == "user":
        print("[ 1 ] Downloading models...")
        download_models()
        print("  Setup complete. Opening the dashboard.")
        print("  Go to Config to define your dimensions and examples.\n")
        launch_ui()
        return

    # ── load-data: just launch UI ────────────────────────────────────────────
    if mode == "load-data":
        launch_ui()
        return

    # ── setup: synthetic data + full pipeline ────────────────────────────────
    if mode == "setup":
        print("[ 1 ] Downloading models...")
        download_models()

        print(f"[ 2 ] Loading shop config (shops/{SHOP}/)...")
        shop_dim_path = Path(paths["dimensions"])
        if not shop_dim_path.exists():
            print(f"\n  ERROR: {shop_dim_path} not found.")
            sys.exit(1)

        from localtailor.config import load_dimensions
        dimensions = load_dimensions()
        print(f"      {len(dimensions)} dimensions: {[d.name for d in dimensions]}\n")

        print("[ 3 ] Generating synthetic dataset...")
        from localtailor.synthetic import generate_synthetic_dataset
        comments_path, gt_path = generate_synthetic_dataset()
        print(f"      Comments → {comments_path}")
        print(f"      Ground truth → {gt_path}\n")

        print("[ 4 ] Cleaning up old models...")
        cleanup_models(paths["models_dir"])

        print("[ 5 ] Training SetFit models...")
        from localtailor.setfit_trainer import train_all_dimensions
        classifiers = train_all_dimensions(dimensions, force=True)

        print("[ 6 ] Running classification pipeline...")
        from localtailor.pipeline import run_pipeline
        predictions_path = run_pipeline(
            comments_path=comments_path,
            classifiers=classifiers,
            dimensions=dimensions,
            post_id="demo",
            output_dir=paths["data_dir"],
        )

        print("[ 7 ] Evaluating against ground truth...")
        from localtailor.evaluator import evaluate
        evaluate(predictions_path, gt_path, post_id="demo",
                 output_dir=paths["data_dir"])

        launch_ui()
        return

    # ── retrain / predict: require existing shop config ──────────────────────
    dim_path = Path(paths["dimensions"])
    if not dim_path.exists() or dim_path.stat().st_size < 10:
        print(f"\n  ERROR: No dimensions found at {dim_path}.")
        print(f"  Check shops/{SHOP}/ or switch SHOP in localtailor/config.py.\n")
        sys.exit(1)

    print("[ 1 ] Loading dimension config...")
    from localtailor.config import load_dimensions
    dimensions = load_dimensions()
    print(f"      {len(dimensions)} dimensions: {[d.name for d in dimensions]}\n")

    # ── Train or load models ─────────────────────────────────────────────────
    if mode == "retrain":
        print("[ 2 ] Cleaning up old models...")
        cleanup_models(paths["models_dir"])

        print("[ 3 ] Training SetFit models...")
        from localtailor.setfit_trainer import train_all_dimensions
        classifiers = train_all_dimensions(dimensions, force=True)
    else:
        print("[ 2 ] Loading existing SetFit models...")
        from localtailor.setfit_trainer import load_all_classifiers
        try:
            classifiers = load_all_classifiers(dimensions)
        except FileNotFoundError as e:
            print(f"\n  ERROR: {e}")
            print("  Run 'retrain' to train models first.\n")
            sys.exit(1)

    # ── Resolve comments path ────────────────────────────────────────────────
    data_dir = paths["data_dir"]
    comments_path = Path(data_dir) / "comments_clean_demo.json"
    gt_path = Path(data_dir) / "ground_truth_demo.json"
    if not comments_path.exists():
        print(f"\n  ERROR: {comments_path} not found.")
        print("  Fetch comments from Facebook or run 'setup' for synthetic data.\n")
        sys.exit(1)

    # retrain uses step 4 onward; predict uses step 3 onward
    step = 4 if mode == "retrain" else 3

    print(f"[ {step} ] Running classification pipeline...")
    from localtailor.pipeline import run_pipeline
    predictions_path = run_pipeline(
        comments_path=comments_path,
        classifiers=classifiers,
        dimensions=dimensions,
        post_id="demo",
        output_dir=data_dir,
    )

    if gt_path.exists():
        print(f"[ {step + 1} ] Evaluating against ground truth...")
        from localtailor.evaluator import evaluate
        evaluate(predictions_path, gt_path, post_id="demo",
                 output_dir=data_dir)

    launch_ui()


if __name__ == "__main__":
    main()
