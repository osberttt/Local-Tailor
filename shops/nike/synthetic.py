"""
shops/nike/synthetic.py
========================
Synthetic Nike shop dataset with 120 comments and ground truth labels.

Comments are modelled on real Nike/Jordan sneaker review patterns and test:
  - Multi-dimension comments (fit + style + comfort, etc.)
  - Short / ambiguous comments
  - Sarcastic comments
  - Questions (intent: needs reply)
  - N/A dimensions (comment mentions one thing, silent on others)
"""

from __future__ import annotations
from typing import Dict, List


# ── Dimensions for this shop ─────────────────────────────────────────────────

ALL_DIMS = ["fit", "comfort", "style", "durability", "price_value", "intent", "tone"]


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
    # ── Fit: runs small ───────────────────────────────────────────────────────
    ("ordered my usual 9.5 and they were way too tight, had to size up to a 10",
     _gt({"fit": ("runs small","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("runs small, definitely go half a size up from your normal",
     _gt({"fit": ("runs small","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("toes were crushed in my regular size, very narrow fit",
     _gt({"fit": ("runs small","classified"), "comfort": ("uncomfortable","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("too snug right out of the box, should have ordered a size larger",
     _gt({"fit": ("runs small","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("runs small compared to other Jordans I own, exchange for bigger size",
     _gt({"fit": ("runs small","classified"), "intent": ("comparison","classified"), "tone": ("neutral","classified")})),

    # ── Fit: true to size ─────────────────────────────────────────────────────
    ("fits perfectly in my usual size, no adjustments needed",
     _gt({"fit": ("true to size","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("true to size, the size chart is accurate",
     _gt({"fit": ("true to size","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("ordered my normal size and they fit like a glove right out of the box",
     _gt({"fit": ("true to size","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("sizing is consistent with other Nikes I own, true to size",
     _gt({"fit": ("true to size","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("fits exactly as expected, no need to size up or down",
     _gt({"fit": ("true to size","classified"), "intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    # ── Fit: runs large ───────────────────────────────────────────────────────
    ("runs big! order at least half a size down",
     _gt({"fit": ("runs large","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("way too roomy in my normal size, went back a half size",
     _gt({"fit": ("runs large","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("heel slips out when walking, ordered too big because they run large",
     _gt({"fit": ("runs large","classified"), "comfort": ("uncomfortable","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("generous sizing, could easily go down half a size",
     _gt({"fit": ("runs large","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("i usually wear a women's 9 but the 9 fit like a 10, runs big",
     _gt({"fit": ("runs large","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    # ── Comfort: very comfortable ─────────────────────────────────────────────
    ("so comfortable I wore them all day without any foot fatigue",
     _gt({"comfort": ("very comfortable","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("great cushioning, my feet feel amazing in these Jordans",
     _gt({"comfort": ("very comfortable","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("most comfortable Jordans I have ever owned, incredible support",
     _gt({"comfort": ("very comfortable","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("worn them all day standing at work and zero foot pain",
     _gt({"comfort": ("very comfortable","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("super comfortable right out of the box, no break-in needed",
     _gt({"comfort": ("very comfortable","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Comfort: average comfort ──────────────────────────────────────────────
    ("comfortable enough but nothing special about the cushioning",
     _gt({"comfort": ("average comfort","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("decent comfort for a lifestyle shoe, not premium but acceptable",
     _gt({"comfort": ("average comfort","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("fine to wear for a few hours, not exceptional",
     _gt({"comfort": ("average comfort","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    # ── Comfort: uncomfortable ────────────────────────────────────────────────
    ("caused blisters on the back of my heel on the first wear",
     _gt({"comfort": ("uncomfortable","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("the insole is completely flat, zero cushioning and very uncomfortable",
     _gt({"comfort": ("uncomfortable","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("feet were aching after just an hour of walking, too uncomfortable",
     _gt({"comfort": ("uncomfortable","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("hard and stiff, needs serious insoles to be wearable",
     _gt({"comfort": ("uncomfortable","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Style: looks great ────────────────────────────────────────────────────
    ("the colorway is fire, already got so many compliments",
     _gt({"style": ("looks great","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("stunning colorway, looks even better in person than in the photos",
     _gt({"style": ("looks great","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("the color blocking is on point, very clean and fresh look",
     _gt({"style": ("looks great","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("beautiful design, matches everything and turns heads everywhere",
     _gt({"style": ("looks great","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("so stylish, by far the best looking shoes I own",
     _gt({"style": ("looks great","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Style: looks different from photos ────────────────────────────────────
    ("the color is completely different from the product photos, very off",
     _gt({"style": ("looks different from photos","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("not what I expected at all, the shade is much darker in person",
     _gt({"style": ("looks different from photos","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("the materials look cheaper in real life than in the advertising images",
     _gt({"style": ("looks different from photos","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Style: looks cheap ────────────────────────────────────────────────────
    ("the finish looks sloppy and cheap for a Nike product",
     _gt({"style": ("looks cheap","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("materials feel and look cheap for the price, not impressed",
     _gt({"style": ("looks cheap","classified"), "price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("looks like a knockoff honestly, not the quality I expect from Jordan",
     _gt({"style": ("looks cheap","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    # ── Durability: lasts well ────────────────────────────────────────────────
    ("worn them for months and they still look brand new, amazing quality",
     _gt({"durability": ("lasts well","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("the sole is still intact after daily wear for six months",
     _gt({"durability": ("lasts well","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("very durable construction, holds up well under heavy use",
     _gt({"durability": ("lasts well","classified"), "intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    ("still looking fresh after dozens of wears, excellent quality",
     _gt({"durability": ("lasts well","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Durability: wears out fast ────────────────────────────────────────────
    ("the sole started separating from the upper after just two months",
     _gt({"durability": ("wears out fast","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("stitching came apart quickly, very poor build quality for Nike",
     _gt({"durability": ("wears out fast","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("the outsole wore down incredibly fast with everyday use",
     _gt({"durability": ("wears out fast","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("creasing badly and leather cracked after just a few weeks",
     _gt({"durability": ("wears out fast","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Durability: too early to tell ─────────────────────────────────────────
    ("just got them, looks great so far but too early to say on durability",
     _gt({"durability": ("too early to tell","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("only worn them twice, will update on how they hold up",
     _gt({"durability": ("too early to tell","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("brand new, first impressions are great but need more time",
     _gt({"durability": ("too early to tell","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    # ── Price: too expensive ──────────────────────────────────────────────────
    ("overpriced for what you actually get, not worth the retail",
     _gt({"price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("110 dollars for this quality is a joke, completely overpriced",
     _gt({"price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("can get similar quality shoes for much less from other brands",
     _gt({"price_value": ("too expensive","classified"), "intent": ("comparison","classified"), "tone": ("neutral","classified")})),

    ("premium Nike price but nowhere near premium quality",
     _gt({"price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Price: good value ─────────────────────────────────────────────────────
    ("reasonable price for what you get, solid value",
     _gt({"price_value": ("good value","classified"), "intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    ("caught them on sale and they are great value at that price",
     _gt({"price_value": ("good value","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("fair retail price given the quality and brand name",
     _gt({"price_value": ("good value","classified"), "intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    # ── Price: worth it ───────────────────────────────────────────────────────
    ("absolutely worth every dollar, best shoes I have bought",
     _gt({"price_value": ("worth it","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("expensive but completely worth it, quality is outstanding",
     _gt({"price_value": ("worth it","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("splurged on these and zero regrets, worth every penny",
     _gt({"price_value": ("worth it","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Multi-dimension ───────────────────────────────────────────────────────
    ("runs big but the colorway is fire, sized down and they look amazing",
     _gt({"fit": ("runs large","classified"), "style": ("looks great","classified"),
          "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("true to size, incredibly comfortable, and the style is clean",
     _gt({"fit": ("true to size","classified"), "comfort": ("very comfortable","classified"),
          "style": ("looks great","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("runs small and causes blisters, very disappointed",
     _gt({"fit": ("runs small","classified"), "comfort": ("uncomfortable","classified"),
          "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("beautiful colorway but runs very large and the quality looks cheap",
     _gt({"fit": ("runs large","classified"), "style": ("looks great","classified"),
          "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("comfortable and stylish but overpriced for what you get",
     _gt({"comfort": ("very comfortable","classified"), "style": ("looks great","classified"),
          "price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("perfect fit, stunning colorway, and completely worth the retail price",
     _gt({"fit": ("true to size","classified"), "style": ("looks great","classified"),
          "price_value": ("worth it","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("fell apart in two months and they were not cheap, very poor value",
     _gt({"durability": ("wears out fast","classified"), "price_value": ("too expensive","classified"),
          "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("the cushioning is excellent and they have lasted well over a year",
     _gt({"comfort": ("very comfortable","classified"), "durability": ("lasts well","classified"),
          "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("looks nothing like the photos and the quality is cheap, returning",
     _gt({"style": ("looks different from photos","classified"),
          "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("runs small so size up, but once you get the right size they are great",
     _gt({"fit": ("runs small","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    # ── Intent: needs reply ───────────────────────────────────────────────────
    ("do these run true to size or should I size up for wide feet?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("are these available in a wide width?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("how does the fit compare to Air Force 1s?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("can these be used for light gym work or are they lifestyle only?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("do they restock sold-out colorways?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("is the sole cushioning good enough for standing all day?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("what is the return policy if they do not fit?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    # ── Intent: comparison ────────────────────────────────────────────────────
    ("more comfortable than the Air Force 1s I had before",
     _gt({"intent": ("comparison","classified"), "comfort": ("very comfortable","classified"), "tone": ("happy","classified")})),

    ("not as comfortable as the Jordan 1 highs, lows have less cushion",
     _gt({"intent": ("comparison","classified"), "comfort": ("average comfort","classified"), "tone": ("neutral","classified")})),

    ("better colorway than last year's version, much cleaner",
     _gt({"intent": ("comparison","classified"), "style": ("looks great","classified"), "tone": ("happy","classified")})),

    ("dunks have better durability than these in my experience",
     _gt({"intent": ("comparison","classified"), "durability": ("wears out fast","classified"), "tone": ("neutral","classified")})),

    # ── Intent: spam ─────────────────────────────────────────────────────────
    ("check out our store for the same colorway at a lower price!",
     _gt({"intent": ("spam","classified"), "tone": ("neutral","classified")})),

    ("dm me for replica Jordans at a fraction of the retail price",
     _gt({"intent": ("spam","classified"), "tone": ("neutral","classified")})),

    ("visit our site for the best sneaker deals this week",
     _gt({"intent": ("spam","classified"), "tone": ("neutral","classified")})),

    # ── Edge cases: sarcasm ───────────────────────────────────────────────────
    ("oh great, the sole fell off after a month — typical Nike quality these days",
     _gt({"durability": ("wears out fast","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("fantastic value — if you enjoy spending 110 dollars on paper-thin soles",
     _gt({"price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("love how the color in person looks nothing like what was advertised",
     _gt({"style": ("looks different from photos","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    # ── Edge cases: very short ────────────────────────────────────────────────
    ("love them!",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("terrible.",
     _gt({"intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("runs big.",
     _gt({"fit": ("runs large","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("fire colorway!",
     _gt({"style": ("looks great","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("not bad.",
     _gt({"intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    ("overpriced.",
     _gt({"price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("neutral","classified")})),

    # ── Neutral / informational ───────────────────────────────────────────────
    ("arrived on time, packaging was intact, shoe as described",
     _gt({"intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("standard Nike quality, does what it is supposed to",
     _gt({"intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("ordered online, received within the week, no issues",
     _gt({"intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("the size I ordered matched the size I received, no issues",
     _gt({"fit": ("true to size","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    # ── Gift / third party ────────────────────────────────────────────────────
    ("bought these as a gift for my daughter and she loves them",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("got them for my son for back to school, perfect for the uniform",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("this is my third pair of Jordan 1 Lows in different colorways",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Tone variations ───────────────────────────────────────────────────────
    ("i am absolutely furious, these fell apart after two weeks of casual wear",
     _gt({"durability": ("wears out fast","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("a little let down if I am honest, expected more cushioning for this price",
     _gt({"comfort": ("average comfort","classified"), "price_value": ("too expensive","classified"),
          "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("wondering if the colorway will grow on me, not sure about it yet",
     _gt({"intent": ("monitor","classified"), "tone": ("curious","classified")})),

    ("delivered on time, shoes exactly as listed, no complaints",
     _gt({"intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("genuinely the best pair of Jordans I have ever worn",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Final varied batch ────────────────────────────────────────────────────
    ("comfortable, stylish, true to size — everything you want in a shoe",
     _gt({"fit": ("true to size","classified"), "comfort": ("very comfortable","classified"),
          "style": ("looks great","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("the leather quality is excellent, very well made for the price",
     _gt({"durability": ("lasts well","classified"), "price_value": ("good value","classified"),
          "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("cute shoe but the color really does look different in person",
     _gt({"style": ("looks different from photos","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("good quality overall but size down, they definitely run large",
     _gt({"fit": ("runs large","classified"), "durability": ("lasts well","classified"),
          "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("i was skeptical but these are honestly worth every cent",
     _gt({"price_value": ("worth it","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("not what i expected from Nike, quality has gone downhill",
     _gt({"intent": ("negative review","classified"), "tone": ("disappointed","classified")})),
]
