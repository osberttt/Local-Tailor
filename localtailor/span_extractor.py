"""
localtailor/span_extractor.py
==============================
Extracts dimension-relevant text spans from comments using extractive QA.
Model: deepset/roberta-base-squad2 (fine-tuned on SQuAD2, handles unanswerable)

For each (comment, dimension) pair:
  Question: "What does this comment say about {dimension}?"
  Context:  the full comment text
  Answer:   the relevant span, or None if not mentioned (→ N/A)

SQuAD2 is trained to return no-answer when the question cannot be answered
from the context, which maps naturally to our N/A case.

Confidence thresholds:
  span_score >= THRESHOLD_SPAN  →  span found, return it
  span_score <  THRESHOLD_SPAN  →  N/A (dimension not mentioned)
"""

from __future__ import annotations
import time
from typing import Optional, Tuple

# Confidence below this → treat as N/A (dimension not mentioned)
THRESHOLD_SPAN = 0.15


class SpanExtractor:
    """Wraps roberta-base-squad2 for dimension span extraction."""

    def __init__(self):
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("Run: pip install transformers torch")

        print("  Loading deepset/roberta-base-squad2 (~500MB, cached after first run)...")
        t0 = time.time()
        self._pipe = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=-1,  # CPU; change to 0 for CUDA
        )
        print(f"  Span extractor loaded in {time.time()-t0:.1f}s")

    def extract(
        self,
        comment: str,
        dimension_name: str,
        dimension_description: Optional[str] = None,
    ) -> Tuple[Optional[str], float]:
        """Extract the span of a comment relevant to a dimension.

        Args:
            comment:               Full comment text.
            dimension_name:        e.g. "comfort", "price_value"
            dimension_description: Optional natural language description of dimension.

        Returns:
            (span, confidence)
              span=None if dimension not mentioned in this comment (N/A)
              confidence is the model's score for the answer span
        """
        self._load()

        # Build a natural question.
        # Using a descriptive question improves recall on short/indirect comments.
        dim_display = dimension_name.replace("_", " ")
        if dimension_description:
            question = f"What does this comment say about {dim_display} ({dimension_description})?"
        else:
            question = f"What does this comment say about {dim_display}?"

        try:
            result = self._pipe(
                question=question,
                context=comment,
                handle_impossible_answer=True,  # SQuAD2: returns score for no-answer
            )
        except Exception as e:
            # On tokenization edge cases (very short text, emoji-only, etc.)
            return None, 0.0

        score = result.get("score", 0.0)
        answer = result.get("answer", "").strip()

        # SQuAD2 no-answer case: answer is empty or score too low
        if not answer or score < THRESHOLD_SPAN:
            return None, score

        return answer, score

    def extract_batch(
        self,
        comments: list[str],
        dimension_name: str,
        dimension_description: Optional[str] = None,
    ) -> list[Tuple[Optional[str], float]]:
        """Extract spans for a list of comments for one dimension."""
        self._load()
        results = []
        for comment in comments:
            span, score = self.extract(comment, dimension_name, dimension_description)
            results.append((span, score))
        return results
