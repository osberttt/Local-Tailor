"""
localtailor/setfit_trainer.py
==============================
Trains one SetFit model per dimension using labeled examples from dimensions.yaml.
Saves each trained model to models/{SHOP}/{dimension_name}/.
Runs inference on extracted spans to produce dimension value + confidence.

SetFit architecture:
  1. Fine-tunes a sentence transformer (all-MiniLM-L6-v2) using contrastive
     learning on positive/negative pairs sampled from the labeled examples.
  2. Trains a logistic regression head on the resulting embeddings.

Training time: ~30 seconds per dimension on CPU with 8 examples per class.
Each model is ~90MB on disk.

Confidence thresholding:
  max_prob >= THRESHOLD_CLASSIFY  →  classified (assign top label)
  max_prob <  THRESHOLD_CLASSIFY  →  Unclear (low confidence, needs review)
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from localtailor.config import DimensionConfig, shop_paths

THRESHOLD_CLASSIFY = 0.35   # below this → Unclear


class SetFitDimensionClassifier:
    """Manages training and inference for a single dimension's SetFit model."""

    def __init__(self, dimension: DimensionConfig, models_dir: str | None = None):
        self.dimension = dimension
        self._models_dir = models_dir or shop_paths()["models_dir"]
        self.model_path = Path(self._models_dir) / dimension.name
        self._model = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, force: bool = False) -> None:
        """Train SetFit on the dimension's labeled examples.

        Args:
            force: Retrain even if a saved model exists.
        """
        if self.model_path.exists() and not force:
            print(f"  [{self.dimension.name}] Model exists, loading from {self.model_path}")
            self._load_model()
            return

        try:
            from setfit import SetFitModel, Trainer, TrainingArguments
            from datasets import Dataset
        except ImportError:
            raise ImportError("Run: pip install setfit datasets")

        examples = self.dimension.all_examples()
        if not examples:
            raise ValueError(f"Dimension '{self.dimension.name}' has no training examples.")

        texts, labels = zip(*examples)
        unique_labels = self.dimension.value_labels()
        label2id = {l: i for i, l in enumerate(unique_labels)}

        # Convert to Dataset
        dataset = Dataset.from_dict({
            "text": list(texts),
            "label": [label2id[l] for l in labels],
        })

        print(f"\n  ── Training SetFit: '{self.dimension.name}' ──────────────────")
        print(f"     Classes : {unique_labels}")
        print(f"     Examples: {len(texts)} total ({len(texts)//len(unique_labels)} per class avg)")

        t0 = time.time()
        model = SetFitModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            labels=unique_labels,
        )

        args = TrainingArguments(
            batch_size=16,
            num_epochs=3,           # more epochs help with few examples per class
            num_iterations=20,      # contrastive pairs per example
            evaluation_strategy="no",
            save_strategy="no",
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
        )

        trainer.train()
        elapsed = time.time() - t0
        print(f"     Trained in {elapsed:.1f}s")

        # Save model
        self.model_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(self.model_path))

        # Save label map alongside model
        meta = {"labels": unique_labels, "label2id": label2id,
                "dimension": self.dimension.name, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(self.model_path / "localtailor_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        self._model = model
        print(f"     Saved → {self.model_path}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, span: str) -> Tuple[str, float, str]:
        """Classify an extracted span.

        Args:
            span: The dimension-relevant text span extracted from a comment.

        Returns:
            (value, confidence, flag)
            flag is "classified" or "unclear"
        """
        if self._model is None:
            self._load_model()

        try:
            # predict_proba returns probabilities for each class
            probs = self._model.predict_proba([span])[0]
            labels = self.dimension.value_labels()

            # probs is a numpy array indexed by label2id order
            prob_dict = {label: float(probs[i]) for i, label in enumerate(labels)}
            top_label = max(prob_dict, key=prob_dict.__getitem__)
            top_prob = prob_dict[top_label]

            if top_prob >= THRESHOLD_CLASSIFY:
                return top_label, round(top_prob, 4), "classified"
            else:
                return "Unclear", round(top_prob, 4), "unclear"

        except Exception as e:
            return "Unclear", 0.0, "unclear"

    def predict_batch(self, spans: List[str]) -> List[Tuple[str, float, str]]:
        """Batch prediction for efficiency."""
        if self._model is None:
            self._load_model()

        if not spans:
            return []

        try:
            probs_batch = self._model.predict_proba(spans)
            labels = self.dimension.value_labels()
            results = []

            for probs in probs_batch:
                prob_dict = {label: float(probs[i]) for i, label in enumerate(labels)}
                top_label = max(prob_dict, key=prob_dict.__getitem__)
                top_prob = prob_dict[top_label]

                if top_prob >= THRESHOLD_CLASSIFY:
                    results.append((top_label, round(top_prob, 4), "classified"))
                else:
                    results.append(("Unclear", round(top_prob, 4), "unclear"))

            return results

        except Exception:
            return [("Unclear", 0.0, "unclear")] * len(spans)

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load a previously saved SetFit model from disk."""
        try:
            from setfit import SetFitModel
        except ImportError:
            raise ImportError("Run: pip install setfit")

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No trained model for '{self.dimension.name}' at {self.model_path}.\n"
                f"Run train() first."
            )

        self._model = SetFitModel.from_pretrained(str(self.model_path))

    def is_trained(self) -> bool:
        return self.model_path.exists()


# ── Top-level train-all helper ────────────────────────────────────────────────

def train_all_dimensions(
    dimensions: List[DimensionConfig],
    force: bool = False,
    models_dir: str | None = None,
) -> Dict[str, SetFitDimensionClassifier]:
    """Train SetFit models for all dimensions. Returns dict of classifiers.

    Args:
        dimensions: List of DimensionConfig objects.
        force:      Retrain all even if models already exist.
        models_dir: Override model output directory.

    Returns:
        {dimension_name: SetFitDimensionClassifier}
    """
    m_dir = models_dir or shop_paths()["models_dir"]
    Path(m_dir).mkdir(parents=True, exist_ok=True)
    classifiers = {}

    print(f"\n{'='*60}")
    print(f"  Training SetFit — {len(dimensions)} dimension(s)")
    print(f"{'='*60}")

    for dim in dimensions:
        clf = SetFitDimensionClassifier(dim, models_dir=m_dir)
        clf.train(force=force)
        classifiers[dim.name] = clf

    print(f"\n  All models trained/loaded.\n")
    return classifiers


def load_all_classifiers(
    dimensions: List[DimensionConfig],
    models_dir: str | None = None,
) -> Dict[str, SetFitDimensionClassifier]:
    """Load pre-trained classifiers without retraining."""
    m_dir = models_dir or shop_paths()["models_dir"]
    classifiers = {}
    for dim in dimensions:
        clf = SetFitDimensionClassifier(dim, models_dir=m_dir)
        if not clf.is_trained():
            raise FileNotFoundError(
                f"No model for '{dim.name}'. Run with --retrain first."
            )
        clf._load_model()
        classifiers[dim.name] = clf
    return classifiers
