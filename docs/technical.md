# Local Tailor — Technical Reference

## Architecture Overview

Local Tailor is a 2-stage NLP pipeline that classifies customer comments along configurable dimensions, then displays results in a Streamlit dashboard.

```
  shops/{SHOP}/
  ├── dimensions.yaml
  ├── examples.json
  └── synthetic.py
         │
         ▼
  comments ──► SpanExtractor ──► SetFitClassifier ──► predictions.json
               (RoBERTa QA)      (per-dimension)          │
                                                           ▼
                                                   Streamlit Dashboard
                                                   ├── Dimension Board
                                                   ├── Intent Queue
                                                   ├── Analytics
                                                   ├── Export
                                                   └── Config Editor
```

### Multi-Shop Architecture

Each shop is a self-contained folder under `shops/`:
- `dimensions.yaml` — dimension definitions (names, values, descriptions)
- `examples.json` — training examples (8 per class recommended)
- `synthetic.py` — synthetic comments with ground truth labels

Switch shops by setting `SHOP` in `localtailor/config.py`. All data, models, and outputs are scoped per shop automatically via `shop_paths()`.

## Pipeline Stages

### Stage 1: Span Extraction (`span_extractor.py`)

**Model**: `deepset/roberta-base-squad2` (~500MB)

For each (comment, dimension) pair, the span extractor frames the task as extractive QA:
- **Question**: "What does this comment say about {dimension} ({description})?"
- **Context**: The full comment text
- **Answer**: The relevant substring, or empty if the dimension isn't mentioned

SQuAD2's "unanswerable question" capability maps directly to the N/A case: when a comment doesn't mention a dimension, the model returns an empty answer with low confidence.

**Threshold**: `THRESHOLD_SPAN = 0.15` — below this, the comment is marked N/A for that dimension.

**Output**: `(span: str | None, confidence: float)`

### Stage 2: Classification (`setfit_trainer.py`)

**Model**: `sentence-transformers/all-MiniLM-L6-v2` (~80MB), fine-tuned per dimension via SetFit

SetFit uses contrastive learning to fine-tune a sentence transformer, then trains a logistic regression head on the embeddings. This approach works well with very few examples (8 per class is sufficient).

**Training parameters**:
- `batch_size=16`
- `num_epochs=1` (SetFit converges fast on few-shot tasks)
- `num_iterations=20` (contrastive pairs generated per example)

**Threshold**: `THRESHOLD_CLASSIFY = 0.50` — below this, the prediction is flagged as "unclear".

**Output**: `(value: str, confidence: float, flag: "classified" | "unclear")`

**Model storage**: `models/{SHOP}/{dimension_name}/` (~90MB per dimension)

## Data Flow

```
run_pipeline.py
  │
  ├─ load_dimensions()               shops/{SHOP}/dimensions.yaml + examples.json → List[DimensionConfig]
  │
  ├─ generate_synthetic_dataset()    → data/{SHOP}/comments_clean_demo.json
  │                                    data/{SHOP}/ground_truth_demo.json
  │
  ├─ train/load SetFit models        → models/{SHOP}/{dim}/
  │
  ├─ run_pipeline()                  → data/{SHOP}/predictions_demo.json
  │
  ├─ evaluate()                      → data/{SHOP}/evaluation_demo.json
  │
  └─ streamlit run app.py            reads all JSON files, renders dashboard
```

## Prediction Schema

Each comment gets a prediction object per dimension:

```json
{
  "c001": {
    "comfort": {
      "value": "too soft",
      "flag": "classified",
      "score": 0.82,
      "span": "too soft and mushy",
      "span_score": 0.74
    }
  }
}
```

**Flag values**:
| Flag | Meaning |
|------|---------|
| `classified` | Confident prediction (score >= 0.50) |
| `unclear` | Low confidence, needs human review |
| `na` | Dimension not mentioned in this comment |

## Evaluation Metrics

`evaluator.py` computes per-dimension:
- **Classification accuracy**: correct / (total classified in both prediction and ground truth)
- **N/A precision**: TP_na / (TP_na + FP_na) — how often a predicted N/A is actually N/A
- **N/A recall**: TP_na / (TP_na + FN_na) — how often an actual N/A is detected
- **Unclear rate**: unclear_count / total — percentage the model isn't confident about
- **Confusion matrix**: per-value breakdown of predictions vs ground truth

## Configuration System

### Shop Selection

Set `SHOP` in `localtailor/config.py` to choose the active shop. The `shop_paths()` function resolves all paths:

```python
from localtailor.config import SHOP, shop_paths

paths = shop_paths()       # uses active SHOP
paths = shop_paths("shoe") # override for a specific shop
# Returns: {shop, dimensions, examples, data_dir, models_dir}
```

### dimensions.yaml (`shops/{SHOP}/dimensions.yaml`)

Defines dimension structure. Each dimension has:
- `name`: Unique identifier (lowercase, underscores)
- `enabled`: Boolean toggle
- `values`: List of possible labels, each with a `label` and optional `description`

Minimum 2 values per dimension. The description is used to build better QA questions for span extraction.

### examples.json (`shops/{SHOP}/examples.json`)

Training examples organized as `{dimension_name: {label: [examples...]}}`. Kept separate from dimensions.yaml because:
- Dimensions are stable (structure rarely changes)
- Examples are frequently edited (iterating on accuracy)
- Separation prevents accidental config breakage when editing examples

## Models & Dependencies

### HuggingFace Models (downloaded on first run)

| Model | Size | Purpose | Cache |
|-------|------|---------|-------|
| `all-MiniLM-L6-v2` | ~80MB | SetFit base sentence transformer | `~/.cache/huggingface` |
| `deepset/roberta-base-squad2` | ~500MB | Span extraction (extractive QA) | `~/.cache/huggingface` |

### Python Dependencies

| Package | Purpose |
|---------|---------|
| `setfit` | Few-shot text classification |
| `sentence-transformers` | Sentence embeddings (SetFit base) |
| `transformers` | Model loading, pipelines |
| `torch` | Backend (CPU-only is fine) |
| `datasets` | HuggingFace dataset utilities |
| `streamlit` | Dashboard UI |
| `pyyaml` | Config loading |
| `pandas` | DataFrames in the dashboard |
| `jinja2` | HTML report templating |
| `fpdf2` | PDF report generation (pure Python, no system deps) |

## Report Generation (`reporter.py`)

### HTML Reports
Uses Jinja2 with `templates/report.html`. The template receives a context dict with metadata, summary, coverage, dimension breakdowns, intent queue, and accuracy data.

### PDF Reports
Uses `fpdf2` (pure Python, no GTK/system dependencies needed). Generates tables and bar charts programmatically. No HTML-to-PDF conversion — the PDF is built directly from the same context dict.

Both formats are saved to `reports/report_{post_id}_{timestamp}.{html,pdf}`.

## Streamlit App (`app.py`)

### Module-level setup
- Adds project root to `sys.path` so `from localtailor.*` imports work
- Loads JSON data files at startup (comments, predictions, evaluation)
- Paths resolve from `shop_paths()` — automatically scoped to the active `SHOP`

### Views

| View | Function | Key features |
|------|----------|-------------|
| Dimension Board | `render_dimension_board()` | Comments grouped by value, highlighted spans, full comment text |
| Intent Queue | `render_intent_queue()` | Priority-sorted comment list, intent/tone badges |
| Analytics | `render_analytics()` | Coverage chart, value distribution, accuracy table, co-occurrence matrix |
| Export | `render_export()` | HTML/PDF generation with download buttons |
| Config | `render_config()` | Edit dimensions.yaml and examples.json from the UI |

### Streamlit Workarounds

**Re-running the app**: Streamlit reruns the entire script on every interaction. All state is ephemeral unless stored in `st.session_state`.

**Port conflicts**: If port 8501 is in use:
```bash
streamlit run localtailor/app.py --server.port 8502
```

**Headless mode** (for servers without a display):
```bash
streamlit run localtailor/app.py --server.headless true
```

**Caching**: Heavy data loads should use `@st.cache_data` to avoid reloading JSON on every interaction.

## Deployment

### Minimal deployment (single machine)

1. Clone the repo
2. `python -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python run_pipeline.py user` (downloads models, opens UI for configuration)
5. Define dimensions and examples in the Config view, then: `python run_pipeline.py retrain`
6. Subsequent runs: `python run_pipeline.py load-data`
7. For developers: `python run_pipeline.py setup` (demo config + synthetic data + full pipeline)

### Cleanup

`python clean.py` removes generated files for the active shop. Use `python clean.py --all` to clean all shops. Neither touches shop definitions (`shops/`), source code, or the virtual environment.

### Server deployment

```bash
streamlit run localtailor/app.py --server.headless true --server.port 8501 --server.address 0.0.0.0
```

### Environment variables

| Variable | Purpose |
|----------|---------|
| `PYTHONIOENCODING=utf-8` | Fixes Unicode errors on Windows |
| `TRANSFORMERS_CACHE` | Override HuggingFace model cache location |
| `CUDA_VISIBLE_DEVICES` | Set to `""` to force CPU, or `"0"` for GPU |

### GPU Support

All models default to CPU (`device=-1`). To use GPU:
1. Install `torch` with CUDA support
2. Edit `span_extractor.py` and change `device=-1` to `device=0`

GPU is not required — CPU inference is fast enough for datasets under ~1000 comments.

## Synthetic Dataset (`synthetic.py`)

Each shop defines its own synthetic comments and ground truth in `shops/{SHOP}/synthetic.py`. The dispatcher in `localtailor/synthetic.py` imports the active shop's module and generates the standard JSON files.

Built-in shops:
- **pillow** — 127 comments across 6 dimensions (comfort, shape, durability, price_value, intent, tone)
- **shoe** — 116 comments across 7 dimensions (fit, comfort, durability, price_value, style, intent, tone)

Comments are designed to test realistic edge cases:
- Multi-dimension comments (one comment covers comfort + price + shape)
- Short/ambiguous comments
- Sarcastic comments
- Questions (intent: needs reply)
- N/A dimensions (comment mentions one thing, silent on others)

Output:
- `data/{SHOP}/comments_clean_demo.json` — comment list with IDs, messages, timestamps, like counts
- `data/{SHOP}/ground_truth_demo.json` — per-comment ground truth for all dimensions

See [Creating a New Shop](creating_a_shop.md) for how to add a new shop with its own synthetic dataset.

## File-by-File Reference

| File | Purpose |
|------|---------|
| `run_pipeline.py` | CLI entry point, orchestrates all steps |
| `localtailor/config.py` | SHOP variable, shop_paths(), DimensionConfig dataclass, YAML/JSON loader |
| `localtailor/synthetic.py` | Dispatcher — imports and runs `shops/{SHOP}/synthetic.py` |
| `localtailor/span_extractor.py` | RoBERTa span extraction |
| `localtailor/setfit_trainer.py` | SetFit training + inference per dimension |
| `localtailor/pipeline.py` | Orchestrates span -> classify |
| `localtailor/evaluator.py` | Accuracy metrics |
| `localtailor/reporter.py` | Jinja2 HTML + fpdf2 PDF report generation |
| `localtailor/app.py` | Streamlit dashboard (5 views) |
| `localtailor/embedder.py` | BERT sentence embeddings (utility, not in main pipeline) |
| `shops/{SHOP}/dimensions.yaml` | Dimension definitions for a shop |
| `shops/{SHOP}/examples.json` | Training examples for a shop |
| `shops/{SHOP}/synthetic.py` | Synthetic comments + ground truth for a shop |
