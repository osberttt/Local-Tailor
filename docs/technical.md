# Technical Reference

## Architecture

```
comments → SpanExtractor (RoBERTa QA) → SetFitClassifier (per-dim) → predictions.json → Streamlit
```

## Stage 1: Span Extraction — `span_extractor.py`

Model: `deepset/roberta-base-squad2` (~500MB, cached in `~/.cache/huggingface`)

Frames each (comment, dimension) as extractive QA:
- Question: "What does this comment say about {dimension} ({description})?"
- Context: full comment text
- SQuAD2's "unanswerable" capability handles the N/A case

Threshold: `THRESHOLD_SPAN = 0.15` — below → N/A for that dimension.
Output: `(span: str | None, confidence: float)`

## Stage 2: Classification — `setfit_trainer.py`

Model: `sentence-transformers/all-MiniLM-L6-v2` (~80MB), fine-tuned per dimension via SetFit.

SetFit: contrastive fine-tuning of sentence transformer → logistic regression head. Works with ~8 examples/class.

Training params: `batch_size=16, num_epochs=1, num_iterations=20`
Threshold: `THRESHOLD_CLASSIFY = 0.50` — below → "unclear"
Output: `(value: str, confidence: float, flag: classified|unclear)`
Storage: `models/{SHOP}/{dimension_name}/` (~90MB each)

## Prediction Schema

```json
{ "c001": { "comfort": { "value": "too soft", "flag": "classified", "score": 0.82, "span": "too soft and mushy", "span_score": 0.74 } } }
```

| Flag | Meaning |
|------|---------|
| `classified` | Confident (score >= 0.50) |
| `unclear` | Low confidence, needs review |
| `na` | Dimension not mentioned |

## Evaluation — `evaluator.py`

Per-dimension metrics:
- Classification accuracy: correct / total classified
- N/A precision/recall
- Unclear rate
- Confusion matrix per value

## Config System

`localtailor/config.py` owns `Shop` enum, `SHOP` variable, `shop_paths()`, `DimensionConfig` dataclass.

```python
SHOP = Shop.SHOE  # set active shop
paths = shop_paths()  # → {shop, dimensions, examples, data_dir, models_dir}
```

**dimensions.yaml**: dimension name, enabled toggle, values (label + description). Min 2 values. Description improves span extraction quality.

**examples.json**: `{ dim_name: { label: [examples...] } }`. Separate from YAML because examples change frequently, structure doesn't.

## Data Flow

```
run_pipeline.py
  → load_dimensions()        shops/{SHOP}/*.yaml + *.json → List[DimensionConfig]
  → generate_synthetic()     → data/{SHOP}/comments_clean_demo.json + ground_truth_demo.json
  → train/load SetFit        → models/{SHOP}/{dim}/
  → run_pipeline()           → data/{SHOP}/predictions_demo.json
  → evaluate()               → data/{SHOP}/evaluation_demo.json
  → streamlit app.py
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `setfit` | Few-shot classification |
| `sentence-transformers` | Embeddings (SetFit base) |
| `transformers` | Model loading, QA pipeline |
| `torch` | Backend (CPU-only fine) |
| `streamlit` | Dashboard |
| `pyyaml` | Config |
| `jinja2` / `fpdf2` | HTML/PDF reports |

## Reports — `reporter.py`

HTML via Jinja2 (`templates/report.html`). PDF via fpdf2 (pure Python, no system deps). Both saved to `reports/report_{post_id}_{timestamp}.{ext}`.

## GPU

All models default to CPU (`device=-1`). For GPU: install torch+CUDA, change `device=-1` to `device=0` in `span_extractor.py`. CPU is fine for <1000 comments.
