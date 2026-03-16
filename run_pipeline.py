"""
Local Tailor — Pipeline Entry Point
=====================================
Usage:
  python run_pipeline.py first-time   # generate synthetic data + train + predict + evaluate + UI
  python run_pipeline.py retrain      # retrain models (after editing dimensions/examples) + predict + UI
  python run_pipeline.py predict      # load existing models + predict on current comments + UI
  python run_pipeline.py load-data    # just launch the Streamlit UI

Key files:
  config/dimensions.yaml   — dimension names, value labels, descriptions (structure only)
  data/examples.json       — training examples per value label (edit to retrain)
"""

import argparse
import subprocess
import sys
from pathlib import Path

MODES = ["first-time", "retrain", "predict", "load-data"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Local Tailor Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Modes:",
            "  first-time   Full setup: generate synthetic data, train models, predict, evaluate, launch UI",
            "  retrain      Retrain all models after editing dimensions/examples, then predict and launch UI",
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
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "localtailor/app.py",
        "--server.headless", "false",
    ])


def main():
    args = parse_args()
    mode = args.mode

    # Ensure output dirs
    for d in ["data", "models"]:
        Path(d).mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("  LOCAL TAILOR")
    print("  Comment intelligence fitted to your business.")
    print("=" * 60 + "\n")

    # ── load-data: just launch UI ──────────────────────────────────────────────
    if mode == "load-data":
        launch_ui()
        return

    # ── Load dimensions ────────────────────────────────────────────────────────
    print("[ 1 ] Loading dimension config...")
    from localtailor.config import load_dimensions
    dimensions = load_dimensions("config/dimensions.yaml")
    print(f"      {len(dimensions)} dimensions: {[d.name for d in dimensions]}\n")

    # ── Generate synthetic dataset (first-time only) ───────────────────────────
    if mode == "first-time":
        print("[ 2 ] Generating synthetic dataset...")
        from localtailor.synthetic import generate_synthetic_dataset
        comments_path, gt_path = generate_synthetic_dataset()
        print(f"      Comments → {comments_path}")
        print(f"      Ground truth → {gt_path}\n")

    # ── Train or load models ───────────────────────────────────────────────────
    if mode in ("first-time", "retrain"):
        print("[ 3 ] Training SetFit models...")
        print("      First run downloads all-MiniLM-L6-v2 (~80MB, cached afterwards)")
        from localtailor.setfit_trainer import train_all_dimensions
        classifiers = train_all_dimensions(dimensions, force=True)
    else:
        # predict mode: load existing models
        print("[ 3 ] Loading existing SetFit models...")
        from localtailor.setfit_trainer import load_all_classifiers
        try:
            classifiers = load_all_classifiers(dimensions)
        except FileNotFoundError as e:
            print(f"\n  ERROR: {e}")
            print("  Run 'first-time' or 'retrain' to train models first.\n")
            sys.exit(1)

    # ── Resolve comments path ──────────────────────────────────────────────────
    if mode != "first-time":
        comments_path = Path("data/comments_clean_demo.json")
        gt_path = Path("data/ground_truth_demo.json")
        if not comments_path.exists():
            print(f"\n  ERROR: {comments_path} not found.")
            print("  Run 'first-time' to generate synthetic data,")
            print("  or place your comments file at that path.\n")
            sys.exit(1)

    # ── Run classification pipeline ────────────────────────────────────────────
    print("[ 4 ] Running classification pipeline...")
    print("      First run downloads roberta-base-squad2 (~500MB)")
    from localtailor.pipeline import run_pipeline
    predictions_path = run_pipeline(
        comments_path=comments_path,
        classifiers=classifiers,
        dimensions=dimensions,
        post_id="demo",
    )

    # ── Evaluate (only if ground truth exists) ─────────────────────────────────
    if gt_path.exists():
        print("[ 5 ] Evaluating against ground truth...")
        from localtailor.evaluator import evaluate
        evaluate(predictions_path, gt_path, post_id="demo")

    # ── Launch UI ──────────────────────────────────────────────────────────────
    launch_ui()


if __name__ == "__main__":
    main()
