# Local Tailor

Classifies Facebook comments by user-defined dimensions using SetFit few-shot models + QA span extraction. Streamlit dashboard for results.

## File Routing

Read only the files for the relevant concern:

| Concern | Files | Doc |
|---------|-------|-----|
| **UI** | `localtailor/app.py` | — |
| **Config** | `localtailor/config.py`, `shops/{shop}/dimensions.yaml`, `shops/{shop}/examples.json` | — |
| **ML Training** | `localtailor/setfit_trainer.py`, `localtailor/embedder.py` | `docs/technical.md` |
| **Pipeline** | `localtailor/pipeline.py`, `localtailor/span_extractor.py` | `docs/technical.md` |
| **Evaluation** | `localtailor/evaluator.py` | `docs/technical.md` |
| **Synthetic Data** | `localtailor/synthetic.py`, `shops/{shop}/synthetic.py` | — |
| **Reports** | `localtailor/reporter.py` | — |
| **New Shop** | `localtailor/config.py`, `shops/` | `docs/creating_a_shop.md` |
| **Facebook API** | — | `docs/facebook_integration.md` |
| **Entry Point** | `run_pipeline.py` | — |

## Commands

```
python run_pipeline.py user       # download models + launch UI
python run_pipeline.py setup      # synthetic + train + predict + eval + UI
python run_pipeline.py retrain    # retrain after config edits + predict + UI
python run_pipeline.py predict    # existing models + predict + UI
python run_pipeline.py load-data  # just launch UI
streamlit run localtailor/app.py  # UI only
```

## Conventions

- Python 3.13, venv at `./venv/`
- Never touch `venv/`, `models/`, `data/`, `__pycache__/` — all generated
- Active shop: `SHOP` in `localtailor/config.py`. Paths resolve via `shop_paths()`
- Layout: `shops/{shop}/` (config), `data/{shop}/` (output), `models/{shop}/` (trained)
- Prediction schema: `{ comment_id: { dim: { value, flag, score, span, span_score } } }`
- Flags: `classified` (score >= 0.50) | `unclear` (low confidence) | `na` (not mentioned)
