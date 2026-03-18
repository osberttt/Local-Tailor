"""
localtailor/evaluator.py
=========================
Measures prediction quality against ground truth labels.

Metrics produced:
  - Per-dimension classification accuracy
  - N/A detection precision/recall
  - Unclear rate (ambiguous comment rate)
  - Example-count sensitivity data (accuracy at 2/4/8 examples per class)
  - Overall weighted accuracy

All results saved to data/evaluation_{post_id}.json
"""

from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def evaluate(
    predictions_path: Path,
    ground_truth_path: Path,
    post_id: str,
    output_dir: str | None = None,
) -> Dict:
    """Compare predictions against ground truth and produce metric report.

    Args:
        predictions_path:  Path to predictions_{post_id}.json
        ground_truth_path: Path to ground_truth_demo.json
        post_id:           Session ID for output filename
        output_dir:        Override output directory (defaults to "data")

    Returns:
        metrics dict (also saved to evaluation_{post_id}.json)
    """
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    gt_lookup: Dict[str, Dict] = {
        item["comment_id"]: item["ground_truth"]
        for item in gt_data["ground_truth"]
    }

    dimensions = list(next(iter(gt_lookup.values())).keys())

    # ── Per-dimension stats ───────────────────────────────────────────────────
    dim_stats = {}
    overall_correct = 0
    overall_total = 0

    for dim in dimensions:
        correct = 0
        total = 0
        na_tp = 0   # predicted N/A, actually N/A
        na_fp = 0   # predicted N/A, actually classified
        na_fn = 0   # predicted classified/unclear, actually N/A
        unclear_count = 0
        confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for cid, gt in gt_lookup.items():
            if cid not in predictions:
                continue

            gt_entry = gt.get(dim, {})
            pred_entry = predictions[cid].get(dim, {})

            gt_value = gt_entry.get("value", "N/A")
            gt_flag = gt_entry.get("flag", "na")
            pred_value = pred_entry.get("value", "N/A")
            pred_flag = pred_entry.get("flag", "na")

            total += 1

            # N/A tracking
            if pred_flag == "na":
                if gt_flag == "na":
                    na_tp += 1
                else:
                    na_fp += 1
            else:
                if gt_flag == "na":
                    na_fn += 1

            # Unclear tracking
            if pred_flag == "unclear":
                unclear_count += 1

            # Value accuracy (only for classified predictions vs classified GT)
            if gt_flag == "classified" and pred_flag == "classified":
                confusion[gt_value][pred_value] += 1
                if pred_value == gt_value:
                    correct += 1
                overall_correct += 1 if pred_value == gt_value else 0
                overall_total += 1

        # N/A precision and recall
        na_precision = na_tp / (na_tp + na_fp) if (na_tp + na_fp) > 0 else 0.0
        na_recall = na_tp / (na_tp + na_fn) if (na_tp + na_fn) > 0 else 0.0

        classified_total = total - na_tp - na_fp  # GT classified items
        accuracy = correct / classified_total if classified_total > 0 else 0.0

        dim_stats[dim] = {
            "accuracy": round(accuracy, 4),
            "accuracy_pct": f"{accuracy*100:.1f}%",
            "correct": correct,
            "classified_total": classified_total,
            "na_precision": round(na_precision, 4),
            "na_recall": round(na_recall, 4),
            "unclear_count": unclear_count,
            "unclear_rate": round(unclear_count / total, 4) if total > 0 else 0.0,
            "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        }

    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0

    metrics = {
        "post_id": post_id,
        "total_comments": len(gt_lookup),
        "dimensions_evaluated": dimensions,
        "overall_accuracy": round(overall_accuracy, 4),
        "overall_accuracy_pct": f"{overall_accuracy*100:.1f}%",
        "per_dimension": dim_stats,
    }

    # Save metrics
    _out_dir = Path(output_dir) if output_dir else Path("data")
    _out_dir.mkdir(parents=True, exist_ok=True)
    output_path = _out_dir / f"evaluation_{post_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    _print_summary(metrics)

    return metrics


def _print_summary(metrics: Dict) -> None:
    print(f"\n  {'='*60}")
    print(f"  EVALUATION RESULTS — {metrics['post_id']}")
    print(f"  {'='*60}")
    print(f"  Overall accuracy : {metrics['overall_accuracy_pct']}")
    print(f"  Total comments   : {metrics['total_comments']}\n")

    for dim, stats in metrics["per_dimension"].items():
        na_pct = f"  N/A precision: {stats['na_precision']*100:.0f}%  recall: {stats['na_recall']*100:.0f}%"
        print(f"  {dim:15s}: {stats['accuracy_pct']:6s} "
              f"({stats['correct']}/{stats['classified_total']} classified correct)  "
              f"unclear: {stats['unclear_count']}  {na_pct}")

    print(f"  {'='*60}\n")


def compute_sensitivity_curve(
    dimensions: List,
    post_id: str = "demo",
    example_counts: List[int] = [2, 4, 8],
) -> Dict:
    """Train SetFit with varying example counts and measure accuracy at each level.
    Returns data suitable for plotting an accuracy curve.

    This is run offline for the hackathon demo chart.
    Saves to data/sensitivity_{post_id}.json.
    """
    from localtailor.synthetic import generate_synthetic_dataset
    from localtailor.setfit_trainer import SetFitDimensionClassifier
    from localtailor.span_extractor import SpanExtractor
    from localtailor.pipeline import run_pipeline
    import copy

    # Ensure synthetic data exists
    comments_path, gt_path = generate_synthetic_dataset()

    with open(gt_path) as f:
        gt_data = json.load(f)

    results = {}

    for n_examples in example_counts:
        print(f"\n  [sensitivity] Training with {n_examples} examples per class...")

        # Clone dimensions with limited examples
        clipped_dims = []
        for dim in dimensions:
            clipped = copy.deepcopy(dim)
            for v in clipped.values:
                v.examples = v.examples[:n_examples]
            clipped_dims.append(clipped)

        # Train
        classifiers = {}
        for dim in clipped_dims:
            clf = SetFitDimensionClassifier(dim)
            clf.train(force=True)
            classifiers[dim.name] = clf

        # Predict
        pred_path = run_pipeline(
            comments_path=comments_path,
            classifiers=classifiers,
            dimensions=clipped_dims,
            post_id=f"sensitivity_{n_examples}",
        )

        # Evaluate
        m = evaluate(pred_path, gt_path, post_id=f"sensitivity_{n_examples}")
        results[n_examples] = {
            "overall_accuracy": m["overall_accuracy"],
            "per_dimension": {
                dim: stats["accuracy"] for dim, stats in m["per_dimension"].items()
            }
        }

    output_path = Path("data") / f"sensitivity_{post_id}.json"
    with open(output_path, "w") as f:
        json.dump({"example_counts": example_counts, "results": results}, f, indent=2)

    print(f"\n  Sensitivity curve saved → {output_path}")
    return results
