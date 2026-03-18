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
python run_pipeline.py setup
```

The dashboard opens at **http://localhost:8501** when done.

**First run downloads ~580MB of models** (cached in `~/.cache/huggingface` afterwards):
- `all-MiniLM-L6-v2` — SetFit base model (~80MB)
- `deepset/roberta-base-squad2` — span extractor (~500MB)

## Multi-Shop Support

Local Tailor supports multiple shops. Each shop has its own dimensions, training examples, and synthetic dataset. Switch between shops by changing one line in `localtailor/config.py`:

```python
SHOP = "pillow"   # switch to "shoe", or any shop you create
```

**Built-in shops**: `pillow` (6 dimensions), `shoe` (7 dimensions)

To create your own shop, see [Creating a New Shop](docs/creating_a_shop.md).

## Common Commands

| What you want to do | Command |
|----------------------|---------|
| New user (download models + open empty UI to configure) | `python run_pipeline.py user` |
| Full setup (synthetic data + train + predict + eval + UI) | `python run_pipeline.py setup` |
| Retrain after editing dimensions/examples | `python run_pipeline.py retrain` |
| Predict on new/updated comments | `python run_pipeline.py predict` |
| Just open the dashboard | `python run_pipeline.py load-data` |
| Clean up active shop's generated files | `python clean.py` |
| Clean up all shops' generated files | `python clean.py --all` |

All commands automatically use the shop set in `localtailor/config.py`.

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
4. Run `python run_pipeline.py retrain` to apply

### Option B: Edit files directly

**Training examples** — `shops/{SHOP}/examples.json`:
```json
{
  "comfort": {
    "too firm": ["this pillow is rock hard", "way too stiff", ...],
    "just right": ["perfect softness", "ideal comfort", ...]
  }
}
```

**Dimension structure** — `shops/{SHOP}/dimensions.yaml`:
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

1. Add a block to `shops/{SHOP}/dimensions.yaml` (min 2 values)
2. Add matching examples to `shops/{SHOP}/examples.json` (8 per class recommended)
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
├── clean.py                       ← delete all generated files and models
├── requirements.txt               ← pip dependencies
├── shops/                         ← shop definitions (one folder per shop)
│   ├── pillow/
│   │   ├── dimensions.yaml        ← pillow shop dimensions
│   │   ├── examples.json          ← pillow shop training examples
│   │   └── synthetic.py           ← pillow shop synthetic comments + ground truth
│   └── shoe/
│       ├── dimensions.yaml        ← shoe shop dimensions
│       ├── examples.json          ← shoe shop training examples
│       └── synthetic.py           ← shoe shop synthetic comments + ground truth
├── data/{SHOP}/                   ← generated data scoped per shop
│   ├── comments_clean_demo.json   ← comments dataset (synthetic or fetched)
│   ├── predictions_demo.json      ← model predictions per comment
│   ├── ground_truth_demo.json     ← ground truth labels (synthetic only)
│   └── evaluation_demo.json       ← accuracy metrics
├── models/{SHOP}/                 ← trained SetFit models (one folder per dimension)
├── reports/                       ← exported HTML/PDF reports
├── templates/
│   └── report.html                ← Jinja2 HTML report template
└── localtailor/
    ├── config.py                  ← SHOP variable, shop_paths(), DimensionConfig loader
    ├── synthetic.py               ← dispatches to shops/{SHOP}/synthetic.py
    ├── embedder.py                ← sentence embeddings utility
    ├── span_extractor.py          ← RoBERTa span extraction
    ├── setfit_trainer.py          ← SetFit train + predict per dimension
    ├── pipeline.py                ← orchestrates span + classify
    ├── evaluator.py               ← accuracy vs ground truth
    ├── reporter.py                ← Jinja2 HTML + fpdf2 PDF generation
    └── app.py                     ← Streamlit dashboard (5 views)
```

## Further Documentation

- [Creating a New Shop](docs/creating_a_shop.md) — step-by-step guide to adding a new shop
- [Technical Reference](docs/technical.md) — architecture, pipeline internals, deployment
- [Facebook Integration](docs/facebook_integration.md) — connecting to Facebook Graph API
