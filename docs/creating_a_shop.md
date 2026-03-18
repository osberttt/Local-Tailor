# Creating a New Shop

## Structure

```
shops/{name}/
├── __init__.py          ← empty, required for imports
├── dimensions.yaml      ← dimension definitions
├── examples.json        ← 8 training examples per class
└── synthetic.py         ← test comments with ground truth
```

Output goes to `data/{name}/` and `models/{name}/` automatically.

## Step 1: dimensions.yaml

Each dimension needs: `name` (lowercase_underscored), `enabled`, `values` (min 2, each with `label` + `description`).

```yaml
dimensions:
  - name: taste
    enabled: true
    values:
      - label: "too bitter"
        description: "The coffee is too bitter, harsh, or strong"
      - label: "just right"
        description: "The taste is balanced, smooth, and enjoyable"
      - label: "too weak"
        description: "The coffee is watery, bland, or lacks flavor"

  # intent and tone are reusable across shops — copy from an existing shop
  - name: intent
    enabled: true
    values:
      - label: "needs reply"
        description: "Comment contains a question awaiting an answer"
      # ... (see shops/shoe/dimensions.yaml for full intent/tone values)
```

Descriptions matter — the span extractor uses them to find relevant text.

## Step 2: examples.json

8 examples per value per dimension. Vary length and phrasing. No overlap between values.

```json
{
  "_readme": "Training examples for SetFit. 8 per class recommended.",
  "taste": {
    "too bitter": [
      "way too bitter, couldn't finish it",
      "the espresso was harsh and burnt tasting",
      "... (8 total)"
    ],
    "just right": ["..."],
    "too weak": ["..."]
  },
  "intent": { "needs reply": ["..."], "...": ["..."] }
}
```

## Step 3: synthetic.py

Test comments with ground truth labels for evaluation. Aim for 80-150 comments.

```python
ALL_DIMS = ["taste", "temperature", "service", "price_value", "intent", "tone"]

def _na(dims):
    return {d: {"value": "N/A", "flag": "na"} for d in dims}

def _gt(overrides):
    gt = _na(ALL_DIMS)
    for dim, (value, flag) in overrides.items():
        gt[dim] = {"value": value, "flag": flag}
    return gt

COMMENTS_RAW = [
    ("this coffee is way too bitter, couldn't even finish it",
     _gt({"taste": ("too bitter","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),
    # ... more comments covering all values, edge cases (sarcasm, multi-dim, short, questions)
]
```

`ALL_DIMS` must match dimension names in dimensions.yaml exactly.

## Step 4: Register the shop

In `localtailor/config.py`:

```python
class Shop(str, Enum):
    PILLOW = "pillow"
    SHOE   = "shoe"
    COFFEE = "coffee"   # ← add

SHOP = Shop.COFFEE      # ← activate
```

## Step 5: Run

```bash
python run_pipeline.py setup
```

## Iteration

- Low accuracy on a value → add more varied examples in examples.json
- Wrong spans extracted → improve `description` in dimensions.yaml
- Too many Unclear → make examples more distinct between values
- Missing edge cases → add more comments to synthetic.py
- After changes: `python run_pipeline.py retrain`

## Checklist

- [ ] `shops/{name}/__init__.py` exists
- [ ] dimensions.yaml: 2+ dimensions, 2+ values each
- [ ] examples.json: 8 examples per value
- [ ] synthetic.py: `ALL_DIMS` and `COMMENTS_RAW` defined, ALL_DIMS matches yaml
- [ ] Shop added to enum + SHOP set in config.py
- [ ] `python run_pipeline.py setup` runs clean

## Built-in Shops

**pillow** — 6 dims (comfort, shape, durability, price_value, intent, tone), 127 comments
**shoe** — 7 dims (fit, comfort, durability, price_value, style, intent, tone), 116 comments
