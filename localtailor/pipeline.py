"""
localtailor/pipeline.py
========================
Orchestrates the full classification pipeline for a set of comments:

  For each comment × dimension:
    1. SpanExtractor  → try to extract relevant span (used for display)
    2. SetFit         → classify span (or full comment if no span found)
    3. Flag decision  → classified | unclear | na

  Span extraction is no longer a hard gate.  If the QA model finds a span,
  we classify that span.  If not, we classify the full comment text.
  A comment is only marked N/A when SetFit itself is uncertain on the full
  comment (clf_score < THRESHOLD_NA_NO_SPAN).

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

# When no span is found, only classify if SetFit is this confident on the full comment.
# Below this → N/A (dimension genuinely not mentioned).
THRESHOLD_NA_NO_SPAN = 0.50


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

    predictions: Dict[str, Dict] = {}
    total_comments = len(comments)

    for dim in dimensions:
        clf = classifiers[dim.name]

        print(f"\n  ── Dimension: '{dim.name}' ──────────────────────────────")
        t0 = time.time()

        stats = {"classified": 0, "unclear": 0, "na": 0}

        for comment in comments:
            cid = comment["id"]
            if cid not in predictions:
                predictions[cid] = {}

            text = comment["message"]

            # Step 1: Try to extract a relevant span (for display).
            # Use only the dimension name — long descriptions confuse the QA model.
            span, span_score = span_extractor.extract(text, dim.name)

            # Step 2: Classify.
            # Use the extracted span when available; fall back to the full comment.
            classify_text = span if span is not None else text
            value, clf_score, flag = clf.predict(classify_text)

            # Step 3: Determine flag.
            # If no span was found, require higher SetFit confidence to call it
            # classified/unclear — otherwise the dimension is not mentioned (N/A).
            if span is None and clf_score < THRESHOLD_NA_NO_SPAN:
                predictions[cid][dim.name] = {
                    "value": "N/A",
                    "flag": "na",
                    "score": round(clf_score, 4),
                    "span": None,
                    "span_score": round(span_score, 4),
                }
                stats["na"] += 1
                continue

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
