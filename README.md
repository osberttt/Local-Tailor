# Local Tailor
*Comment intelligence fitted to your business.*

## Quick Start

```bash
# 1. Create virtual environment (first time only)
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline + open the dashboard
python run_pipeline.py
```

The dashboard opens at **http://localhost:8501** when done.

**First run downloads ~580MB of models** (cached in `~/.cache/huggingface` afterwards):
- `all-MiniLM-L6-v2` — SetFit base model (~80MB)
- `deepset/roberta-base-squad2` — span extractor (~500MB)

## Common Commands

| What you want to do | Command |
|----------------------|---------|
| First time setup (generate data + train + predict + UI) | `python run_pipeline.py first-time` |
| Retrain after editing dimensions/examples | `python run_pipeline.py retrain` |
| Predict on new/updated comments | `python run_pipeline.py predict` |
| Just open the dashboard | `python run_pipeline.py load-data` |

## Dashboard Views

| View | What it shows |
|------|--------------|
| **Dimension Board** | Comments grouped by dimension value, with highlighted relevant spans |
| **Intent Queue** | Comments sorted by action priority (unanswered questions first) |
| **Analytics** | Coverage charts, value distribution, cross-dimension co-occurrence |
| **Export** | Generate downloadable HTML and/or PDF reports |
| **Config** | Edit dimensions, values, and training examples from the UI |

## Editing Dimensions & Examples

### Option A: From the dashboard
1. Go to **Config** in the sidebar
2. Add/remove dimensions, values, and examples
3. Click **Save changes**
4. Run `python run_pipeline.py --retrain` to apply

### Option B: Edit files directly

**Training examples** — `data/examples.json`:
```json
{
  "comfort": {
    "too firm": ["this pillow is rock hard", "way too stiff", ...],
    "just right": ["perfect softness", "ideal comfort", ...]
  }
}
```

**Dimension structure** — `config/dimensions.yaml`:
```yaml
dimensions:
  - name: comfort
    enabled: true
    values:
      - label: "too firm"
        description: "The pillow is too hard or stiff"
      - label: "just right"
        description: "Comfort level is perfect"
```

Then retrain:
```bash
python run_pipeline.py retrain
```

SetFit retrains in ~30 seconds per dimension on CPU.

## Adding a New Dimension

1. Add a block to `config/dimensions.yaml` (min 2 values)
2. Add matching examples to `data/examples.json` (8 per class recommended)
3. Run `python run_pipeline.py retrain`

Or use the **Config** view in the dashboard to do all three steps visually.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Make sure your venv is activated: `venv\Scripts\activate` |
| Dashboard won't open | Run manually: `streamlit run localtailor/app.py` |
| Port 8501 already in use | Kill the old process, or use `streamlit run localtailor/app.py --server.port 8502` |
| Models re-downloading every run | Check `~/.cache/huggingface` exists and has space |
| Training examples not taking effect | You must run `python run_pipeline.py retrain` after editing examples |
| `UnicodeEncodeError` on Windows | Set environment variable: `set PYTHONIOENCODING=utf-8` |

## File Structure

```
local-tailor/
├── run_pipeline.py                ← entry point (pipeline + UI launcher)
├── requirements.txt               ← pip dependencies
├── config/
│   └── dimensions.yaml            ← dimension names, values, descriptions
├── data/
│   ├── examples.json              ← training examples (edit to retrain)
│   ├── comments_clean_demo.json   ← cleaned comments dataset
│   ├── predictions_demo.json      ← model predictions per comment
│   ├── ground_truth_demo.json     ← synthetic ground truth labels
│   └── evaluation_demo.json       ← accuracy metrics
├── models/                        ← trained SetFit models (one folder per dimension)
├── reports/                       ← exported HTML/PDF reports
├── templates/
│   └── report.html                ← Jinja2 HTML report template
└── localtailor/
    ├── config.py                  ← DimensionConfig + YAML/JSON loader
    ├── synthetic.py               ← synthetic demo dataset generator
    ├── embedder.py                ← sentence embeddings utility
    ├── span_extractor.py          ← RoBERTa span extraction
    ├── setfit_trainer.py          ← SetFit train + predict per dimension
    ├── pipeline.py                ← orchestrates span + classify
    ├── evaluator.py               ← accuracy vs ground truth + sensitivity
    ├── reporter.py                ← Jinja2 HTML + fpdf2 PDF generation
    └── app.py                     ← Streamlit dashboard (5 views)
```

## Further Documentation

- [User Guide](docs/user_guide.md) — step-by-step instructions for non-technical users
- [Technical Reference](docs/technical.md) — architecture, pipeline internals, deployment
