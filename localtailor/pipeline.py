"""
localtailor/pipeline.py
========================
Orchestrates the full classification pipeline for a set of comments:

  For each comment × dimension:
    1. SpanExtractor  → extract relevant span (or None → N/A)
    2. SetFit         → classify span → value + confidence

Output: predictions_{post_id}.json
Schema:
  {
    "c001": {
      "comfort": {
        "value":     "too soft",
        "flag":      "classified",     # classified | unclear | na
        "score":     0.82,
        "span":      "too soft and mushy",
        "span_score": 0.74
      },
      ...
    }
  }
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List

from localtailor.config import DimensionConfig
from localtailor.span_extractor import SpanExtractor
from localtailor.setfit_trainer import SetFitDimensionClassifier


def run_pipeline(
    comments_path: Path,
    classifiers: Dict[str, SetFitDimensionClassifier],
    dimensions: List[DimensionConfig],
    post_id: str,
    output_dir: str | None = None,
) -> Path:
    """Run full span→classify pipeline on all comments.

    Args:
        comments_path: Path to comments_clean_{post_id}.json
        classifiers:   Dict of {dim_name: SetFitDimensionClassifier}
        dimensions:    List of DimensionConfig (for descriptions)
        post_id:       Session ID for output filenames
        output_dir:    Override output directory (defaults to "data")

    Returns:
        Path to predictions_{post_id}.json
    """
    with open(comments_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    comments = data["comments"]
    print(f"\n  Comments to classify : {len(comments)}")
    print(f"  Dimensions           : {[d.name for d in dimensions]}")

    # Load span extractor once (shared across all dims)
    span_extractor = SpanExtractor()

    # Build dim_name → description from value descriptions
    # Combines all value descriptions into a general dimension description
    dim_general_desc = {}
    for d in dimensions:
        descs = [v.description for v in d.values if v.description]
        if descs:
            dim_general_desc[d.name] = "; ".join(descs[:3])
        else:
            dim_general_desc[d.name] = d.name.replace("_", " ")

    predictions: Dict[str, Dict] = {}
    total_comments = len(comments)

    for dim in dimensions:
        desc = dim_general_desc.get(dim.name, dim.name.replace("_", " "))
        clf = classifiers[dim.name]

        print(f"\n  ── Dimension: '{dim.name}' ──────────────────────────────")
        t0 = time.time()

        stats = {"classified": 0, "unclear": 0, "na": 0}

        for comment in comments:
            cid = comment["id"]
            if cid not in predictions:
                predictions[cid] = {}

            text = comment["message"]

            # Step 1: Extract span
            span, span_score = span_extractor.extract(text, dim.name, desc)

            if span is None:
                # Dimension not mentioned in this comment
                predictions[cid][dim.name] = {
                    "value": "N/A",
                    "flag": "na",
                    "score": round(span_score, 4),
                    "span": None,
                    "span_score": round(span_score, 4),
                }
                stats["na"] += 1
                continue

            # Step 2: Classify span
            value, clf_score, flag = clf.predict(span)

            predictions[cid][dim.name] = {
                "value": value,
                "flag": flag,
                "score": clf_score,
                "span": span,
                "span_score": round(span_score, 4),
            }
            stats[flag] += 1

        elapsed = time.time() - t0
        pct = lambda k: f"{stats[k]/total_comments*100:.0f}%"
        print(f"     classified : {stats['classified']:3d} ({pct('classified')})")
        print(f"     unclear    : {stats['unclear']:3d} ({pct('unclear')})")
        print(f"     N/A        : {stats['na']:3d} ({pct('na')})")
        print(f"     time       : {elapsed:.1f}s")

    # Save predictions
    _out_dir = Path(output_dir) if output_dir else Path("data")
    _out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = _out_dir / f"predictions_{post_id}.json"

    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\n  Predictions saved → {pred_path}\n")
    return pred_path
