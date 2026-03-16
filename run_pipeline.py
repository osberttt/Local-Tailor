"""
Local Tailor — Pipeline Entry Point
=====================================
Usage:
  python run_pipeline.py user         # download models + launch empty UI for user to configure
  python run_pipeline.py setup        # copy demo config + synthetic data + train + predict + eval + UI
  python run_pipeline.py retrain      # retrain models after editing dimensions/examples + predict + UI
  python run_pipeline.py predict      # load existing models + predict on current comments + UI
  python run_pipeline.py load-data    # just launch the Streamlit UI

Files:
  config/dimensions.yaml   — user's dimension config (created via UI or setup)
  data/examples.json       — user's training examples (created via UI or setup)
  demo/dimensions.yaml     — dev/demo pillow shop dimensions (read-only reference)
  demo/examples.json       — dev/demo pillow shop examples (read-only reference)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MODES = ["user", "setup", "retrain", "predict", "load-data"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Local Tailor Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Modes:",
            "  user         Download models, launch empty UI for initial configuration",
            "  setup        Copy demo config, generate synthetic data, train, predict, evaluate, launch UI",
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
        # Check for a key file that every model has
        result = try_to_load_from_cache(model_name, "config.json")
        return isinstance(result, str)  # returns path string if cached, None or sentinel otherwise
    except Exception:
        return False


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

    # Ensure output dirs
    for d in ["data", "models", "config"]:
        Path(d).mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("  LOCAL TAILOR")
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

    # ── load-data: just launch UI ──────────────────────────────────────────────
    if mode == "load-data":
        launch_ui()
        return

    # ── setup: copy demo config + full pipeline ──────────────────────────────
    if mode == "setup":
        print("[ 1 ] Downloading models...")
        download_models()

        print("[ 2 ] Loading demo config (demo/ → config/ + data/)...")
        demo_dir = Path("demo")
        if not (demo_dir / "dimensions.yaml").exists():
            print("\n  ERROR: demo/dimensions.yaml not found.")
            sys.exit(1)
        shutil.copy(demo_dir / "dimensions.yaml", "config/dimensions.yaml")
        shutil.copy(demo_dir / "examples.json", "data/examples.json")
        print("      Copied demo/dimensions.yaml → config/dimensions.yaml")
        print("      Copied demo/examples.json   → data/examples.json\n")

        from localtailor.config import load_dimensions
        dimensions = load_dimensions("config/dimensions.yaml")
        print(f"      {len(dimensions)} dimensions: {[d.name for d in dimensions]}\n")

        print("[ 3 ] Generating synthetic dataset...")
        from localtailor.synthetic import generate_synthetic_dataset
        comments_path, gt_path = generate_synthetic_dataset()
        print(f"      Comments → {comments_path}")
        print(f"      Ground truth → {gt_path}\n")

        print("[ 4 ] Training SetFit models...")
        from localtailor.setfit_trainer import train_all_dimensions
        classifiers = train_all_dimensions(dimensions, force=True)

        print("[ 5 ] Running classification pipeline...")
        from localtailor.pipeline import run_pipeline
        predictions_path = run_pipeline(
            comments_path=comments_path,
            classifiers=classifiers,
            dimensions=dimensions,
            post_id="demo",
        )

        print("[ 6 ] Evaluating against ground truth...")
        from localtailor.evaluator import evaluate
        evaluate(predictions_path, gt_path, post_id="demo")

        launch_ui()
        return

    # ── retrain / predict: require existing user config ────────────────────────
    config_path = Path("config/dimensions.yaml")
    if not config_path.exists() or config_path.stat().st_size < 10:
        print("\n  ERROR: No dimensions found in config/dimensions.yaml.")
        print("  Run 'user' to configure, or 'setup' for the dev dataset.\n")
        sys.exit(1)

    print("[ 1 ] Loading dimension config...")
    from localtailor.config import load_dimensions
    dimensions = load_dimensions("config/dimensions.yaml")
    print(f"      {len(dimensions)} dimensions: {[d.name for d in dimensions]}\n")

    # ── Train or load models ───────────────────────────────────────────────────
    if mode == "retrain":
        print("[ 2 ] Training SetFit models...")
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

    # ── Resolve comments path ──────────────────────────────────────────────────
    comments_path = Path("data/comments_clean_demo.json")
    gt_path = Path("data/ground_truth_demo.json")
    if not comments_path.exists():
        print(f"\n  ERROR: {comments_path} not found.")
        print("  Fetch comments from Facebook or run 'setup' for synthetic data.\n")
        sys.exit(1)

    # ── Run classification pipeline ────────────────────────────────────────────
    print("[ 3 ] Running classification pipeline...")
    from localtailor.pipeline import run_pipeline
    predictions_path = run_pipeline(
        comments_path=comments_path,
        classifiers=classifiers,
        dimensions=dimensions,
        post_id="demo",
    )

    # ── Evaluate (only if ground truth exists) ─────────────────────────────────
    if gt_path.exists():
        print("[ 4 ] Evaluating against ground truth...")
        from localtailor.evaluator import evaluate
        evaluate(predictions_path, gt_path, post_id="demo")

    launch_ui()


if __name__ == "__main__":
    main()
