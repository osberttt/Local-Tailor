# Creating a New Shop

This guide walks through adding a new shop to Local Tailor, using a **coffee shop** as an example.

## Overview

Each shop is a self-contained folder under `shops/` with three files:

```
shops/coffee/
├── dimensions.yaml    ← what dimensions to classify (taste, temperature, ...)
├── examples.json      ← 8 training examples per class
└── synthetic.py       ← synthetic comments with ground truth labels
```

When you set `SHOP = "coffee"` in `localtailor/config.py` and run the pipeline, all data and models are scoped automatically:
- Synthetic data and predictions go to `data/coffee/`
- Trained models go to `models/coffee/`

## Step 1: Create the shop folder

Create the directory and an empty `__init__.py` (required for Python imports):

```bash
mkdir shops/coffee
touch shops/coffee/__init__.py
```

## Step 2: Define dimensions (`dimensions.yaml`)

Think about what your customers talk about when reviewing your product. Each dimension should have:
- A **name** (lowercase, underscores, unique)
- At least **2 values** (the possible labels)
- A **description** per value (helps the span extractor understand what to look for)

Create `shops/coffee/dimensions.yaml`:

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

  - name: temperature
    enabled: true
    values:
      - label: "too hot"
        description: "The drink was served too hot, burned the tongue"
      - label: "just right"
        description: "The temperature was perfect for drinking"
      - label: "too cold"
        description: "The drink was lukewarm or cold when served"

  - name: service
    enabled: true
    values:
      - label: "great service"
        description: "Staff was friendly, fast, or helpful"
      - label: "slow service"
        description: "Long wait times or inattentive staff"
      - label: "rude service"
        description: "Staff was impolite, dismissive, or unfriendly"

  - name: price_value
    enabled: true
    values:
      - label: "too expensive"
        description: "The drink is overpriced for what you get"
      - label: "good value"
        description: "Fair price for the quality"
      - label: "worth it"
        description: "Premium but justified by quality"

  # intent and tone are common across most shops
  - name: intent
    enabled: true
    values:
      - label: "needs reply"
        description: "Comment contains a question awaiting an answer"
      - label: "positive review"
        description: "Compliment or positive experience"
      - label: "negative review"
        description: "Complaint or negative experience"
      - label: "comparison"
        description: "Compares to another shop or product"
      - label: "monitor"
        description: "General discussion worth watching"
      - label: "spam"
        description: "Off-topic or promotional comment"

  - name: tone
    enabled: true
    values:
      - label: "happy"
        description: "Enthusiastic or clearly satisfied"
      - label: "curious"
        description: "Asking questions or uncertain"
      - label: "disappointed"
        description: "Let down or quietly dissatisfied"
      - label: "angry"
        description: "Frustrated or strongly negative"
      - label: "neutral"
        description: "Factual or no strong emotion"
```

**Tips:**
- **intent** and **tone** are reusable across shops — copy them as-is
- Product-specific dimensions (taste, temperature, service) are what make your shop unique
- Keep value labels short and distinct — they become classifier labels
- Descriptions matter — they're used by the span extractor to find relevant text

## Step 3: Write training examples (`examples.json`)

For each dimension and value, provide **8 example sentences** that a customer might write. These are the training data for SetFit.

Create `shops/coffee/examples.json`:

```json
{
  "_readme": "Training examples for SetFit classifiers. 8 per class recommended.",
  "_format": "{ dimension_name: { value_label: [ example1, example2, ... ] } }",
  "_recommended_per_class": 8,

  "taste": {
    "too bitter": [
      "way too bitter, couldn't finish it",
      "the espresso was harsh and burnt tasting",
      "extremely strong and bitter, not enjoyable",
      "tasted like burnt rubber, way too intense",
      "the bitterness was overwhelming",
      "couldn't drink it without adding three sugars",
      "the roast was way too dark for my taste",
      "bitter aftertaste that lingered for hours"
    ],
    "just right": [
      "perfectly balanced flavor, smooth and rich",
      "the taste is amazing, best coffee in town",
      "smooth with just the right amount of strength",
      "excellent roast, not too strong not too weak",
      "the flavor profile is exactly what I want",
      "perfectly brewed, can taste the quality",
      "rich and smooth, no bitterness at all",
      "the best tasting latte I've had in months"
    ],
    "too weak": [
      "tastes like hot water with a hint of coffee",
      "way too watery, barely any coffee flavor",
      "the weakest latte I've ever had",
      "no body to the coffee at all, very thin",
      "couldn't taste any coffee through the milk",
      "disappointingly weak, expected more punch",
      "bland and flavorless, like drinking warm water",
      "needs at least a double shot to taste like coffee"
    ]
  },

  "temperature": {
    "too hot": [
      "burned my tongue, way too hot to drink",
      "had to wait 20 minutes before I could sip it",
      "scalding hot, almost burned myself",
      "the cup was so hot I couldn't hold it",
      "way too hot, should have let it cool first",
      "burned my mouth on the first sip",
      "dangerously hot, nearly spilled it",
      "the temperature was extreme, couldn't drink it right away"
    ],
    "just right": [
      "perfect drinking temperature right away",
      "served at the ideal temperature, could sip immediately",
      "the temperature was spot on, warm and comfortable",
      "great temp, didn't have to wait at all",
      "perfectly warm, enjoyed every sip",
      "the right temperature from the first sip to the last",
      "warm enough to enjoy, not so hot it burns",
      "ideal serving temperature, very impressed"
    ],
    "too cold": [
      "my coffee was barely warm when I got it",
      "lukewarm at best, very disappointing",
      "the latte was cold by the time I got it",
      "not hot enough, felt like room temperature",
      "the coffee was cold, had to ask them to remake it",
      "barely warm, definitely not freshly made",
      "arrived cold, probably sat on the counter too long",
      "tepid coffee is unacceptable at these prices"
    ]
  },

  "service": {
    "great service": [
      "the barista was so friendly and helpful",
      "fast service, had my drink in under two minutes",
      "staff was incredibly welcoming and attentive",
      "the barista remembered my usual order, love that",
      "quick and efficient, great customer service",
      "friendly staff who really care about quality",
      "excellent service from start to finish",
      "the team here is always cheerful and fast"
    ],
    "slow service": [
      "waited 15 minutes for a simple black coffee",
      "the line barely moved, way too slow",
      "took forever to get my order, very frustrating",
      "slow service even when the shop was empty",
      "had to ask twice about my order, they forgot",
      "the wait time here is ridiculous",
      "my coffee took so long it was cold when I got it",
      "painfully slow, won't come back during rush hour"
    ],
    "rude service": [
      "the barista was incredibly rude to me",
      "got attitude when I asked to remake my order",
      "staff was dismissive and unfriendly",
      "the cashier rolled their eyes when I asked a question",
      "very rude service, felt unwelcome",
      "the barista acted like I was bothering them",
      "terrible attitude from the staff today",
      "felt judged for ordering a simple coffee"
    ]
  },

  "price_value": {
    "too expensive": [
      "way overpriced for a basic coffee",
      "can't justify paying this much for a latte",
      "highway robbery, there are cheaper options everywhere",
      "the prices here are ridiculous for what you get",
      "too expensive for the quality served",
      "I can make better coffee at home for a fraction of the cost",
      "the price keeps going up but the quality doesn't",
      "not worth the premium price they charge"
    ],
    "good value": [
      "fair price for a decent cup of coffee",
      "reasonable prices and good quality",
      "solid coffee at a sensible price point",
      "good value, especially the lunch combo",
      "affordable for the quality you get",
      "the prices are fair and the coffee is good",
      "decent deal, no complaints about the price",
      "one of the more affordable specialty coffee shops"
    ],
    "worth it": [
      "expensive but the quality is outstanding",
      "absolutely worth every baht for this quality",
      "the best coffee in the area, worth the premium",
      "you get what you pay for and this delivers",
      "worth the splurge, exceptional craftsmanship",
      "premium price but premium quality to match",
      "I'd pay even more for coffee this good",
      "worth it for the experience alone"
    ]
  },

  "intent": {
    "needs reply": [
      "do you have oat milk as an option?",
      "what time do you close on weekends?",
      "can I bring my dog inside the shop?",
      "do you offer a loyalty card or rewards program?",
      "is there wifi available for customers?",
      "what beans do you use for the espresso?",
      "do you have any sugar-free syrup options?",
      "can I order online for pickup?"
    ],
    "positive review": [
      "best coffee shop in the neighborhood",
      "absolutely love this place, my daily ritual",
      "five stars, the coffee is consistently amazing",
      "highly recommend, the vibes and coffee are perfect",
      "this shop never disappoints, always great",
      "my favorite coffee spot, keep doing what you do",
      "amazing coffee, amazing atmosphere",
      "the best latte I've ever had, hands down"
    ],
    "negative review": [
      "terrible coffee, will not be coming back",
      "worst coffee experience I've had this year",
      "do not waste your money here",
      "deeply disappointed with the quality",
      "avoid this place, overpriced and underwhelming",
      "the coffee was awful and the service was worse",
      "would give zero stars if I could",
      "complete waste of money, go somewhere else"
    ],
    "comparison": [
      "much better than the Starbucks down the road",
      "not as good as the cafe on the other street",
      "similar to the old shop that used to be here",
      "better coffee but worse atmosphere than the competition",
      "I've tried five shops and this is the best",
      "the other branch is better, this one is inconsistent",
      "way better value than the chain cafes",
      "not quite as good as I expected after all the hype"
    ],
    "monitor": [
      "interesting new menu items this month",
      "been seeing a lot of people posting about this place",
      "curious what the new seasonal blend tastes like",
      "might try this place after reading the comments",
      "the reviews are very mixed on this shop",
      "seems popular with the locals",
      "thinking about checking this out this weekend",
      "the new renovation looks nice from outside"
    ],
    "spam": [
      "check out my coffee blog for more reviews",
      "follow me for daily cafe recommendations",
      "DM me for wholesale coffee bean deals",
      "visit our shop for better coffee at lower prices",
      "totally unrelated but our bakery has a sale",
      "nice coffee! also check out our new roastery",
      "great post, also our shop does free delivery",
      "we sell premium beans at half the price DM us"
    ]
  },

  "tone": {
    "happy": [
      "absolutely love this place, makes my day every time!",
      "so happy I found this gem, best coffee ever",
      "thrilled with the quality, couldn't be happier",
      "this coffee just makes everything better",
      "beyond satisfied, it's my happy place",
      "delighted every single time I visit",
      "the joy of a perfect cup, amazing",
      "genuinely makes my mornings worth waking up for"
    ],
    "curious": [
      "wondering what roast they use for the house blend?",
      "has anyone tried the new cold brew here?",
      "not sure if I should try the pour-over or espresso",
      "is this place busy on Saturday mornings?",
      "thinking about trying the seasonal special",
      "does the quality vary between baristas?",
      "curious if they do coffee classes or tastings",
      "anyone know if they source their beans locally?"
    ],
    "disappointed": [
      "expected much better based on the reviews",
      "a bit let down, it was just average",
      "sadly not as good as last time I visited",
      "underwhelming for the reputation this place has",
      "thought it would be special but it was ordinary",
      "the hype didn't match the reality",
      "not quite what I was hoping for",
      "disappointed after hearing so many good things"
    ],
    "angry": [
      "absolutely furious about the service today",
      "this is a rip-off, worst coffee ever",
      "unacceptable quality at these prices",
      "I'm outraged, they got my order wrong twice",
      "disgusted by the hygiene in this place",
      "never coming back, waste of time and money",
      "furious that I waited 30 minutes for cold coffee",
      "this place is a scam, avoid at all costs"
    ],
    "neutral": [
      "standard coffee shop, nothing remarkable",
      "it's fine, does the job for a morning coffee",
      "average coffee, average prices, average atmosphere",
      "came for a quick coffee, got what I expected",
      "nothing special but nothing wrong either",
      "a basic coffee shop, no complaints",
      "ordered a latte, it was okay",
      "decent enough for a quick stop"
    ]
  }
}
```

**Tips:**
- Each example should sound like a real customer comment
- Vary the length — mix short ("way too bitter") with longer sentences
- Include different ways of saying the same thing
- Don't overlap between values — "burnt tasting" should only appear in "too bitter", not "just right"
- 8 examples is the sweet spot for SetFit — fewer works but accuracy drops

## Step 4: Write synthetic comments (`synthetic.py`)

This file defines test comments with ground truth labels, used for evaluation. Each comment gets labeled across all dimensions.

Create `shops/coffee/synthetic.py`:

```python
"""
shops/coffee/synthetic.py
==========================
Synthetic coffee shop dataset with ground truth labels.
"""

from __future__ import annotations
from typing import Dict, List


ALL_DIMS = ["taste", "temperature", "service", "price_value", "intent", "tone"]


def _na(dims: List[str]) -> Dict:
    return {d: {"value": "N/A", "flag": "na"} for d in dims}

def _gt(overrides: Dict) -> Dict:
    """Build ground truth: start with N/A for all dims, apply overrides."""
    gt = _na(ALL_DIMS)
    for dim, (value, flag) in overrides.items():
        gt[dim] = {"value": value, "flag": flag}
    return gt


COMMENTS_RAW = [
    # ── Taste: too bitter ─────────────────────────────────────────────
    ("this coffee is way too bitter, couldn't even finish it",
     _gt({"taste": ("too bitter","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("love the atmosphere but the espresso is harsh and bitter",
     _gt({"taste": ("too bitter","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Taste: just right ─────────────────────────────────────────────
    ("perfectly balanced latte, smooth and rich",
     _gt({"taste": ("just right","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Multi-dimension ───────────────────────────────────────────────
    ("great taste but the service was painfully slow and overpriced",
     _gt({"taste": ("just right","classified"), "service": ("slow service","classified"),
          "price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Intent: needs reply ───────────────────────────────────────────
    ("do you offer oat milk? my daughter is lactose intolerant",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    # Add more comments following this pattern...
    # Aim for 80-150 comments covering all dimension values
]
```

**Key rules:**
- `ALL_DIMS` must list every dimension name from your `dimensions.yaml`
- Every comment gets ground truth for ALL dimensions — use `_gt()` which defaults unmentioned dimensions to N/A
- Only label dimensions that the comment actually mentions
- Include edge cases: sarcasm, very short comments, multi-dimension comments, questions
- Aim for at least 5 comments per value per dimension

## Step 5: Register and activate the shop

Edit `localtailor/config.py`:

1. Add your shop to the `Shop` enum:

```python
class Shop(str, Enum):
    PILLOW = "pillow"
    SHOE   = "shoe"
    COFFEE = "coffee"   # ← add this
```

2. Set it as the active shop:

```python
SHOP = Shop.COFFEE
```

## Step 6: Run the pipeline

```bash
python run_pipeline.py setup
```

This will:
1. Download base models (if not cached)
2. Load your dimensions and examples from `shops/coffee/`
3. Generate synthetic comments from `shops/coffee/synthetic.py` into `data/coffee/`
4. Train SetFit models and save to `models/coffee/`
5. Run classification on synthetic comments
6. Evaluate against ground truth
7. Launch the Streamlit dashboard

## Step 7: Iterate

After the first run, look at the evaluation results and improve:

- **Low accuracy on a value?** Add more varied examples in `examples.json`
- **Wrong spans being extracted?** Improve the `description` field in `dimensions.yaml` — the span extractor uses it
- **Too many Unclear predictions?** The examples may be too similar between values — make them more distinct
- **Synthetic comments missing edge cases?** Add more comments to `synthetic.py`

Retrain after changes:
```bash
python run_pipeline.py retrain
```

## Checklist

- [ ] `shops/{name}/__init__.py` exists (empty file)
- [ ] `shops/{name}/dimensions.yaml` has at least 2 dimensions with 2+ values each
- [ ] `shops/{name}/examples.json` has 8 examples per value per dimension
- [ ] `shops/{name}/synthetic.py` defines `ALL_DIMS` and `COMMENTS_RAW`
- [ ] `ALL_DIMS` in synthetic.py matches dimension names in dimensions.yaml
- [ ] Shop added to `Shop` enum in `localtailor/config.py`
- [ ] `SHOP` in `localtailor/config.py` is set to your shop
- [ ] `python run_pipeline.py setup` runs without errors

## Reference: Built-in Shops

### Pillow shop (`shops/pillow/`)

| Dimension | Values |
|-----------|--------|
| comfort | too firm, just right, too soft, changes over time |
| shape | too thin, just right thickness, too thick, loses shape |
| durability | lasts well, degrades quickly, too early to tell |
| price_value | too expensive, good value, worth it |
| intent | needs reply, positive review, negative review, comparison, monitor, spam |
| tone | happy, curious, disappointed, angry, neutral |

127 synthetic comments.

### Shoe shop (`shops/shoe/`)

| Dimension | Values |
|-----------|--------|
| fit | too tight, true to size, too loose, breaks in |
| comfort | very comfortable, average comfort, uncomfortable |
| durability | lasts well, wears out fast, too early to tell |
| price_value | too expensive, good value, worth it |
| style | looks great, looks different, looks cheap |
| intent | needs reply, positive review, negative review, comparison, monitor, spam |
| tone | happy, curious, disappointed, angry, neutral |

116 synthetic comments.
