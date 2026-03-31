"""
shops/womens_clothing/synthetic.py
===================================
Synthetic women's clothing shop dataset with 130 comments and ground truth labels.

Comments are modelled on real e-commerce clothing review patterns and test:
  - Multi-dimension comments (fit + material + price, etc.)
  - Sarcastic / indirect comments
  - Short / ambiguous comments
  - Questions (intent: needs reply)
  - N/A dimensions (comment mentions one thing, silent on others)
"""

from __future__ import annotations
from typing import Dict, List


# ── Dimensions for this shop ─────────────────────────────────────────────────

ALL_DIMS = ["sizing", "material", "style", "comfort", "price_value", "intent", "tone"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _na(dims: List[str]) -> Dict:
    return {d: {"value": "N/A", "flag": "na"} for d in dims}

def _gt(overrides: Dict) -> Dict:
    """Build ground truth: start with N/A for all dims, apply overrides."""
    gt = _na(ALL_DIMS)
    for dim, (value, flag) in overrides.items():
        gt[dim] = {"value": value, "flag": flag}
    return gt


# ── Comment definitions ──────────────────────────────────────────────────────

COMMENTS_RAW = [
    # ── Sizing: runs small ────────────────────────────────────────────────────
    ("ordered my usual size 6 and it was outrageously tight, had to size up twice",
     _gt({"sizing": ("runs small","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("runs very small, go at least one size up from what you normally wear",
     _gt({"sizing": ("runs small","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("tiny! i'm usually a small and the medium barely fit me",
     _gt({"sizing": ("runs small","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("the petite small was outrageously small, could not even zip it up",
     _gt({"sizing": ("runs small","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("tight across the bust in my normal size, had to reorder larger",
     _gt({"sizing": ("runs small","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    # ── Sizing: true to size ──────────────────────────────────────────────────
    ("fits perfectly in my usual size, no need to size up",
     _gt({"sizing": ("true to size","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("ordered my normal medium and it fits exactly as expected",
     _gt({"sizing": ("true to size","classified"), "intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    ("true to size, the measurements match the size chart perfectly",
     _gt({"sizing": ("true to size","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("size small fits just right, consistent with this brand's sizing",
     _gt({"sizing": ("true to size","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("bought my usual size and it fits beautifully, tts",
     _gt({"sizing": ("true to size","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Sizing: runs large ────────────────────────────────────────────────────
    ("way too big even in the smallest size they offer",
     _gt({"sizing": ("runs large","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("i am swimming in this small, definitely runs large",
     _gt({"sizing": ("runs large","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("ordered a medium and it hangs off me like a tent, go down a size",
     _gt({"sizing": ("runs large","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("runs very large, size down from your usual",
     _gt({"sizing": ("runs large","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("the xs is still too roomy for my frame, very generously cut",
     _gt({"sizing": ("runs large","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    # ── Material: great quality ───────────────────────────────────────────────
    ("the fabric is beautiful, thick and luxurious, clearly well-made",
     _gt({"material": ("great quality","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("beautifully lined and constructed, the seams are clean and sturdy",
     _gt({"material": ("great quality","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("this fabric drapes so beautifully, you can tell the quality is premium",
     _gt({"material": ("great quality","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("washed it three times and it still looks brand new, excellent quality",
     _gt({"material": ("great quality","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("the construction is excellent, very impressed with the quality",
     _gt({"material": ("great quality","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Material: poor quality ────────────────────────────────────────────────
    ("the fabric feels incredibly cheap and thin, very disappointed",
     _gt({"material": ("poor quality","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("scratchy and rough against the skin, terrible material",
     _gt({"material": ("poor quality","classified"), "comfort": ("uncomfortable","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("seams came apart after the very first wash, awful construction",
     _gt({"material": ("poor quality","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("the fabric is see-through and looks very cheap in person",
     _gt({"material": ("poor quality","classified"), "style": ("looks different from photos","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("pilled terribly after just two washes, very poor material quality",
     _gt({"material": ("poor quality","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Material: too early to tell ───────────────────────────────────────────
    ("just received it, initial feel is good but need more wears to judge",
     _gt({"material": ("too early to tell","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("only worn it once so far, seems like decent quality but will see",
     _gt({"material": ("too early to tell","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("brand new, looks and feels fine but i will update after washing",
     _gt({"material": ("too early to tell","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    # ── Style: flattering ─────────────────────────────────────────────────────
    ("so flattering on my figure, got so many compliments the first time i wore it",
     _gt({"style": ("flattering","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("the cut is incredibly flattering for an hourglass shape",
     _gt({"style": ("flattering","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("makes my waist look tiny and the rest of my figure look amazing",
     _gt({"style": ("flattering","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("very flattering silhouette, i feel great wearing it",
     _gt({"style": ("flattering","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("enhances my shape so well, the most flattering dress i own",
     _gt({"style": ("flattering","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Style: unflattering ───────────────────────────────────────────────────
    ("the cut does absolutely nothing for my figure, very boxy and unflattering",
     _gt({"style": ("unflattering","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("overwhelmed my small frame completely, looked ridiculous on me",
     _gt({"style": ("unflattering","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("made me look short and squat, not a good silhouette at all",
     _gt({"style": ("unflattering","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("the waist falls in entirely the wrong place for my body shape",
     _gt({"style": ("unflattering","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Style: looks different from photos ────────────────────────────────────
    ("looks nothing like the photo, very misleading product images",
     _gt({"style": ("looks different from photos","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("the color in person is completely different from what was pictured",
     _gt({"style": ("looks different from photos","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("in real life it looks much cheaper than it does online",
     _gt({"style": ("looks different from photos","classified"), "material": ("poor quality","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("the print looks totally different in person, very disappointed",
     _gt({"style": ("looks different from photos","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Comfort: very comfortable ─────────────────────────────────────────────
    ("so comfortable i forget i am wearing it, could live in this",
     _gt({"comfort": ("very comfortable","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("incredibly soft against the skin, so comfortable all day",
     _gt({"comfort": ("very comfortable","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("the most comfortable dress i own, i wear it constantly",
     _gt({"comfort": ("very comfortable","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("so cozy and comfortable, feels amazing on",
     _gt({"comfort": ("very comfortable","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Comfort: average comfort ──────────────────────────────────────────────
    ("comfortable enough but nothing particularly special about how it feels",
     _gt({"comfort": ("average comfort","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("decent comfort for everyday wear, no complaints but nothing wow",
     _gt({"comfort": ("average comfort","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("fine to wear, comfort is acceptable for the type of garment",
     _gt({"comfort": ("average comfort","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    # ── Comfort: uncomfortable ────────────────────────────────────────────────
    ("the fabric is so itchy i cannot wear it for more than an hour",
     _gt({"comfort": ("uncomfortable","classified"), "material": ("poor quality","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("so tight and restrictive, can barely move in it",
     _gt({"sizing": ("runs small","classified"), "comfort": ("uncomfortable","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("the lining is scratchy and makes it very unpleasant to wear all day",
     _gt({"comfort": ("uncomfortable","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("digs in uncomfortably at the waist after a couple of hours",
     _gt({"comfort": ("uncomfortable","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Price: too expensive ──────────────────────────────────────────────────
    ("way overpriced for the quality you actually receive",
     _gt({"price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("i expected so much better for this price point, not worth it at all",
     _gt({"price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("premium price but definitely not premium quality, very disappointing",
     _gt({"price_value": ("too expensive","classified"), "material": ("poor quality","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("can get the same thing elsewhere for a fraction of the cost",
     _gt({"price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("neutral","classified")})),

    # ── Price: good value ─────────────────────────────────────────────────────
    ("reasonable price for the quality, no complaints",
     _gt({"price_value": ("good value","classified"), "intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    ("good value for money, happy with what i paid",
     _gt({"price_value": ("good value","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("solid deal at this price point, quality is proportionate to cost",
     _gt({"price_value": ("good value","classified"), "intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    # ── Price: worth it ───────────────────────────────────────────────────────
    ("absolutely worth every penny, the quality is outstanding",
     _gt({"price_value": ("worth it","classified"), "material": ("great quality","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("yes it is expensive but you get exactly what you pay for",
     _gt({"price_value": ("worth it","classified"), "intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    ("splurged on this and zero regrets, completely worth the investment",
     _gt({"price_value": ("worth it","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Multi-dimension ───────────────────────────────────────────────────────
    ("flattering cut but runs very small, had to order two sizes up",
     _gt({"sizing": ("runs small","classified"), "style": ("flattering","classified"),
          "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("beautiful quality but way overpriced for what you get",
     _gt({"material": ("great quality","classified"), "price_value": ("too expensive","classified"),
          "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("true to size, incredibly comfortable, and worth every cent",
     _gt({"sizing": ("true to size","classified"), "comfort": ("very comfortable","classified"),
          "price_value": ("worth it","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("runs large and the fabric feels cheap, very disappointed in this purchase",
     _gt({"sizing": ("runs large","classified"), "material": ("poor quality","classified"),
          "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("the style is very flattering but the material is disappointingly thin",
     _gt({"style": ("flattering","classified"), "material": ("poor quality","classified"),
          "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("fits like a glove and makes me look amazing, definitely worth the price",
     _gt({"sizing": ("true to size","classified"), "style": ("flattering","classified"),
          "price_value": ("worth it","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("comfortable and flattering but it runs small so size up",
     _gt({"sizing": ("runs small","classified"), "style": ("flattering","classified"),
          "comfort": ("very comfortable","classified"), "intent": ("monitor","classified"), "tone": ("happy","classified")})),

    ("the quality is excellent but unfortunately it runs very large on me",
     _gt({"sizing": ("runs large","classified"), "material": ("great quality","classified"),
          "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("overpriced and uncomfortable, the worst of both worlds",
     _gt({"price_value": ("too expensive","classified"), "comfort": ("uncomfortable","classified"),
          "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("gorgeous fabric, flattering fit, reasonably priced — perfect purchase",
     _gt({"material": ("great quality","classified"), "style": ("flattering","classified"),
          "price_value": ("good value","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Intent: needs reply ───────────────────────────────────────────────────
    ("does this come in a petite length?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("is this available in plus sizes or just standard?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("what is the fabric content of this blouse?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("can this be machine washed or is it dry clean only?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("should i size up if i have a larger bust?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("is the color true to the photos or does it look different in person?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("how does this look on a petite frame?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("is this lined or unlined?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    # ── Intent: comparison ────────────────────────────────────────────────────
    ("much better quality than the similar version they sold last season",
     _gt({"intent": ("comparison","classified"), "material": ("great quality","classified"), "tone": ("happy","classified")})),

    ("not as good as the one i returned last month, that one fit better",
     _gt({"intent": ("comparison","classified"), "tone": ("disappointed","classified")})),

    ("comparable quality to a designer brand but at a fraction of the price",
     _gt({"intent": ("comparison","classified"), "price_value": ("good value","classified"), "tone": ("happy","classified")})),

    ("tried five similar dresses before this and this is by far the best",
     _gt({"intent": ("comparison","classified"), "tone": ("happy","classified")})),

    # ── Intent: spam ─────────────────────────────────────────────────────────
    ("check out my store for the same style at lower prices!",
     _gt({"intent": ("spam","classified"), "tone": ("neutral","classified")})),

    ("visit our website for great deals on women's fashion this weekend",
     _gt({"intent": ("spam","classified"), "tone": ("neutral","classified")})),

    ("dm me for discount codes on all clothing items",
     _gt({"intent": ("spam","classified"), "tone": ("neutral","classified")})),

    # ── Edge cases: sarcasm ───────────────────────────────────────────────────
    ("oh sure, totally worth the price — if you enjoy wearing a paper bag",
     _gt({"material": ("poor quality","classified"), "price_value": ("too expensive","classified"),
          "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("great quality! only fell apart after two wears!",
     _gt({"material": ("poor quality","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("fantastic fit — if your name is Barbie and you have no hips",
     _gt({"style": ("unflattering","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    # ── Edge cases: very short ────────────────────────────────────────────────
    ("love it!",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("terrible.",
     _gt({"intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("runs small.",
     _gt({"sizing": ("runs small","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("perfect!",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("not bad.",
     _gt({"intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    ("overpriced.",
     _gt({"price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("neutral","classified")})),

    ("very flattering!",
     _gt({"style": ("flattering","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Neutral / informational ───────────────────────────────────────────────
    ("arrived on time, packaging was intact, item as described",
     _gt({"intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("standard quality, does what it needs to do without any issues",
     _gt({"intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("the dimensions match the product listing, no surprises",
     _gt({"sizing": ("true to size","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("used it a few times, nothing exceptional to report either way",
     _gt({"intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    # ── Mixed sentiment ───────────────────────────────────────────────────────
    ("the style is gorgeous but the material is disappointingly thin for the price",
     _gt({"style": ("flattering","classified"), "material": ("poor quality","classified"),
          "price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("comfortable and flattering but the quality won't last, sadly",
     _gt({"comfort": ("very comfortable","classified"), "style": ("flattering","classified"),
          "material": ("poor quality","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("the fit is perfect but it is expensive and will probably need dry cleaning",
     _gt({"sizing": ("true to size","classified"), "price_value": ("too expensive","classified"),
          "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("great quality but runs very large and looks different from the photos",
     _gt({"sizing": ("runs large","classified"), "material": ("great quality","classified"),
          "style": ("looks different from photos","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Tone: angry ───────────────────────────────────────────────────────────
    ("i am absolutely furious, this is not what i ordered at all",
     _gt({"intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("outrageous quality for this price, demanding a full refund",
     _gt({"material": ("poor quality","classified"), "price_value": ("too expensive","classified"),
          "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    # ── Tone: curious ─────────────────────────────────────────────────────────
    ("not sure if i should keep this or return it, the sizing is confusing",
     _gt({"intent": ("monitor","classified"), "tone": ("curious","classified")})),

    ("wondering if the color will fade after a few washes, time will tell",
     _gt({"material": ("too early to tell","classified"), "intent": ("monitor","classified"), "tone": ("curious","classified")})),

    # ── Body type / petite / tall notes ──────────────────────────────────────
    ("perfect for tall women, the length is just right for a 5'10 frame",
     _gt({"sizing": ("true to size","classified"), "style": ("flattering","classified"),
          "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("if you are petite this will swamp you, not suitable for short frames",
     _gt({"sizing": ("runs large","classified"), "style": ("unflattering","classified"),
          "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("the skirt is very long, great for tall ladies but petites should be wary",
     _gt({"intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    # ── Repeat purchase / gift ────────────────────────────────────────────────
    ("bought two in different colors because i love it so much",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("bought this as a gift and she was thrilled with it",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("this is my third order of this exact item, that should say everything",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Final varied batch ────────────────────────────────────────────────────
    ("the fabric softens beautifully after washing and keeps its shape",
     _gt({"material": ("great quality","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("very flattering and comfortable, best dress i have bought in years",
     _gt({"style": ("flattering","classified"), "comfort": ("very comfortable","classified"),
          "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("it does not look anything like the model photo, very misleading",
     _gt({"style": ("looks different from photos","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("perfect fit and amazing quality, completely worth the splurge",
     _gt({"sizing": ("true to size","classified"), "material": ("great quality","classified"),
          "price_value": ("worth it","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("a little disappointed if i am honest, expected more for the price",
     _gt({"price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("delivered quickly, item as described, no issues whatsoever",
     _gt({"intent": ("monitor","classified"), "tone": ("neutral","classified")})),
]
