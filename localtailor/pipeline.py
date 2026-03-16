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
) -> Path:
    """Run full span→classify pipeline on all comments.

    Args:
        comments_path: Path to comments_clean_{post_id}.json
        classifiers:   Dict of {dim_name: SetFitDimensionClassifier}
        dimensions:    List of DimensionConfig (for descriptions)
        post_id:       Session ID for output filenames

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

    # Build dim_name → description map for better span extraction questions
    dim_descriptions = {
        d.name: d.values[0].description  # use first value's description as hint
        for d in dimensions if d.values
    }
    # Better: use a general dimension description
    dim_general_desc = {
        "comfort": "how comfortable or supportive the pillow feels",
        "shape": "the shape, thickness, height or loft of the pillow",
        "durability": "how long the pillow lasts or holds up over time",
        "price_value": "the price, cost or value for money",
        "intent": "what the commenter is trying to do or ask",
        "tone": "the emotional tone or feeling of the comment",
    }

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
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    pred_path = output_dir / f"predictions_{post_id}.json"

    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\n  Predictions saved → {pred_path}\n")
    return pred_path
