"""
shops/shoe/synthetic.py
========================
Synthetic shoe shop dataset with ~120 comments and ground truth labels.

Comments test realistic edge cases:
  - Multi-dimension comments (one comment covers fit + comfort + price)
  - Short/ambiguous comments
  - Sarcastic comments
  - Questions (intent: needs reply)
  - N/A dimensions (comment mentions one thing, silent on others)
"""

from __future__ import annotations
from typing import Dict, List


# ── Dimensions for this shop ─────────────────────────────────────────────────

ALL_DIMS = ["fit", "comfort", "durability", "price_value", "style", "intent", "tone"]


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
    # ── Fit: too tight ────────────────────────────────────────────────────────
    ("these shoes squeeze my toes like crazy, way too narrow",
     _gt({"fit": ("too tight","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("ordered my normal size and they're painfully tight",
     _gt({"fit": ("too tight","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("the toe box is ridiculously small, had to return them",
     _gt({"fit": ("too tight","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("way too narrow for my wide feet, couldn't even get them on",
     _gt({"fit": ("too tight","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("constricting across the top, very uncomfortable fit",
     _gt({"fit": ("too tight","classified"), "comfort": ("uncomfortable","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Fit: true to size ─────────────────────────────────────────────────────
    ("perfect fit, ordered my usual size and they're spot on",
     _gt({"fit": ("true to size","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("true to size, no issues with the fit at all",
     _gt({"fit": ("true to size","classified"), "intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    ("fits like a glove, exactly what I expected",
     _gt({"fit": ("true to size","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("sizing is accurate, comfortable right out of the box",
     _gt({"fit": ("true to size","classified"), "comfort": ("very comfortable","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("great fit, matches the size chart perfectly",
     _gt({"fit": ("true to size","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Fit: too loose ────────────────────────────────────────────────────────
    ("way too big, my feet slide around inside them",
     _gt({"fit": ("too loose","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("the heel slips out every time I take a step",
     _gt({"fit": ("too loose","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("runs super large, should have gone a full size down",
     _gt({"fit": ("too loose","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("so loose I need double socks just to keep them on",
     _gt({"fit": ("too loose","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Fit: breaks in ────────────────────────────────────────────────────────
    ("tight at first but after a week of wearing they moulded to my feet perfectly",
     _gt({"fit": ("breaks in","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("stiff for the first few days but they loosened up nicely",
     _gt({"fit": ("breaks in","classified"), "intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    ("give them a few days, the leather stretches and they fit great",
     _gt({"fit": ("breaks in","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("first two wears were painful but now they're the comfiest shoes I own",
     _gt({"fit": ("breaks in","classified"), "comfort": ("very comfortable","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Comfort: very comfortable ─────────────────────────────────────────────
    ("like walking on clouds, my feet feel amazing all day long",
     _gt({"comfort": ("very comfortable","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("the cushioning is incredible, walked 15km without any pain",
     _gt({"comfort": ("very comfortable","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("most comfortable shoes I've ever worn, hands down",
     _gt({"comfort": ("very comfortable","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("the arch support is perfect, my feet love these shoes",
     _gt({"comfort": ("very comfortable","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("can stand all day at work without any foot fatigue",
     _gt({"comfort": ("very comfortable","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Comfort: average comfort ──────────────────────────────────────────────
    ("they're okay for walking around, nothing special though",
     _gt({"comfort": ("average comfort","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("decent comfort for the price, not the best not the worst",
     _gt({"comfort": ("average comfort","classified"), "price_value": ("good value","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("fine for short walks but my feet get tired on longer ones",
     _gt({"comfort": ("average comfort","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    # ── Comfort: uncomfortable ────────────────────────────────────────────────
    ("got massive blisters after just one hour of wearing these",
     _gt({"comfort": ("uncomfortable","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("zero cushioning, it's like walking on bare concrete",
     _gt({"comfort": ("uncomfortable","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("my feet were in agony after a short walk, terrible comfort",
     _gt({"comfort": ("uncomfortable","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("the sole is rock hard, absolutely no padding whatsoever",
     _gt({"comfort": ("uncomfortable","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("rubbed my heels raw on the first day, painful",
     _gt({"comfort": ("uncomfortable","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    # ── Durability: lasts well ────────────────────────────────────────────────
    ("had these for over a year and they still look brand new",
     _gt({"durability": ("lasts well","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("survived rain, mud, and daily commuting, incredibly tough",
     _gt({"durability": ("lasts well","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("the sole shows barely any wear after months of daily use",
     _gt({"durability": ("lasts well","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("excellent build quality, these are made to last",
     _gt({"durability": ("lasts well","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Durability: wears out fast ────────────────────────────────────────────
    ("the sole came unglued after just two months of normal wear",
     _gt({"durability": ("wears out fast","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("stitching started coming apart within the first three weeks",
     _gt({"durability": ("wears out fast","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("the rubber on the sole wore through in under two months",
     _gt({"durability": ("wears out fast","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("fell apart embarrassingly fast, terrible construction",
     _gt({"durability": ("wears out fast","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    # ── Durability: too early to tell ─────────────────────────────────────────
    ("just unboxed them today, they look solid though",
     _gt({"durability": ("too early to tell","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("only worn them twice, too soon to comment on durability",
     _gt({"durability": ("too early to tell","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("brand new, will come back in a few months with an update",
     _gt({"durability": ("too early to tell","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    # ── Price: too expensive ──────────────────────────────────────────────────
    ("way overpriced for what you actually get, not impressed",
     _gt({"price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("I've had better shoes for a fraction of this price",
     _gt({"price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("premium price but the quality is anything but premium",
     _gt({"price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("can't justify spending this much on shoes that fell apart",
     _gt({"price_value": ("too expensive","classified"), "durability": ("wears out fast","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    # ── Price: good value ─────────────────────────────────────────────────────
    ("solid shoes for the price, no complaints at all",
     _gt({"price_value": ("good value","classified"), "intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    ("reasonable price for decent quality, happy with the deal",
     _gt({"price_value": ("good value","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("affordable and they do the job well, can't ask for more",
     _gt({"price_value": ("good value","classified"), "intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    # ── Price: worth it ───────────────────────────────────────────────────────
    ("expensive but absolutely worth it, my feet have never been happier",
     _gt({"price_value": ("worth it","classified"), "comfort": ("very comfortable","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("the best investment in footwear I've ever made",
     _gt({"price_value": ("worth it","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("worth every single penny, exceptional quality",
     _gt({"price_value": ("worth it","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Style: looks great ────────────────────────────────────────────────────
    ("these shoes look absolutely stunning, get compliments daily",
     _gt({"style": ("looks great","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("the design is clean and sleek, love the colour",
     _gt({"style": ("looks great","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("everyone keeps asking where I got these, they look amazing",
     _gt({"style": ("looks great","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("the photos don't do them justice, even better in person",
     _gt({"style": ("looks great","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Style: looks different ────────────────────────────────────────────────
    ("the colour is quite different from what's shown online",
     _gt({"style": ("looks different","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("not exactly what I expected from the photos but still okay",
     _gt({"style": ("looks different","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("the shape looks different in person compared to the listing",
     _gt({"style": ("looks different","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Style: looks cheap ────────────────────────────────────────────────────
    ("the material looks plasticky and cheap, very poor finish",
     _gt({"style": ("looks cheap","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("glue visible everywhere, looks like a knockoff honestly",
     _gt({"style": ("looks cheap","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("scuff marks right out of the box, terrible presentation",
     _gt({"style": ("looks cheap","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    # ── Multi-dimension comments ──────────────────────────────────────────────
    ("great fit and super comfortable but honestly too expensive",
     _gt({"fit": ("true to size","classified"), "comfort": ("very comfortable","classified"),
          "price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("tight fit, ugly design, and the sole fell off after a month",
     _gt({"fit": ("too tight","classified"), "style": ("looks cheap","classified"),
          "durability": ("wears out fast","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("comfortable, stylish, and actually affordable — the perfect combo",
     _gt({"comfort": ("very comfortable","classified"), "style": ("looks great","classified"),
          "price_value": ("good value","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("look great but after two months the sole is already wearing down",
     _gt({"style": ("looks great","classified"), "durability": ("wears out fast","classified"),
          "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("true to size and the build quality is exceptional, will last years",
     _gt({"fit": ("true to size","classified"), "durability": ("lasts well","classified"),
          "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("uncomfortable and overpriced, double disappointment",
     _gt({"comfort": ("uncomfortable","classified"), "price_value": ("too expensive","classified"),
          "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("the fit is perfect and the comfort is amazing, worth every penny",
     _gt({"fit": ("true to size","classified"), "comfort": ("very comfortable","classified"),
          "price_value": ("worth it","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("looks cheap but surprisingly comfortable for the price",
     _gt({"style": ("looks cheap","classified"), "comfort": ("very comfortable","classified"),
          "price_value": ("good value","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    # ── Intent: needs reply ───────────────────────────────────────────────────
    ("do these come in wide width options?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("what size should I get if I'm between a 9 and a 10?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("are these waterproof or just water resistant?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("can I use these for running or are they just for casual wear?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("do you ship to Southeast Asia and how long does it take?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("is the sole replaceable when it eventually wears out?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("what's your return policy if the fit isn't right?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("does the leather stretch over time or should I size up now?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    # ── Intent: comparison ────────────────────────────────────────────────────
    ("much better cushioning than my Nike pair at a similar price",
     _gt({"intent": ("comparison","classified"), "comfort": ("very comfortable","classified"), "tone": ("happy","classified")})),

    ("not as durable as my old Adidas, those lasted three years",
     _gt({"intent": ("comparison","classified"), "durability": ("wears out fast","classified"), "tone": ("disappointed","classified")})),

    ("tried Puma, Nike, and Asics before these and these are the best",
     _gt({"intent": ("comparison","classified"), "tone": ("happy","classified")})),

    ("feels completely different to my last pair, in a good way",
     _gt({"intent": ("comparison","classified"), "tone": ("happy","classified")})),

    # ── Intent: spam ──────────────────────────────────────────────────────────
    ("check out my page for the best shoe deals guaranteed!",
     _gt({"intent": ("spam","classified"), "tone": ("neutral","classified")})),

    ("follow me for daily sneaker reviews and unboxing videos",
     _gt({"intent": ("spam","classified"), "tone": ("neutral","classified")})),

    ("nice shoes! also our store has 50% off all footwear this weekend",
     _gt({"intent": ("spam","classified"), "tone": ("neutral","classified")})),

    # ── Edge cases: sarcasm ───────────────────────────────────────────────────
    ("oh great, nothing like blisters on day one to start the relationship",
     _gt({"comfort": ("uncomfortable","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("love paying premium prices for shoes that fall apart in weeks",
     _gt({"price_value": ("too expensive","classified"), "durability": ("wears out fast","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("fantastic, my heel slips out with every single step, great design",
     _gt({"fit": ("too loose","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    # ── Edge cases: very short ────────────────────────────────────────────────
    ("love them!",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("terrible.",
     _gt({"intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("too tight.",
     _gt({"fit": ("too tight","classified"), "intent": ("negative review","classified"), "tone": ("neutral","classified")})),

    ("perfect fit!",
     _gt({"fit": ("true to size","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("overpriced.",
     _gt({"price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("neutral","classified")})),

    ("so comfy!",
     _gt({"comfort": ("very comfortable","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Edge cases: indirect ──────────────────────────────────────────────────
    ("my dog tried to chew these and couldn't even dent them, that's quality",
     _gt({"durability": ("lasts well","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("bought a pair for my dad and now he wants another colour",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("wore these to a wedding and got more compliments than the bride",
     _gt({"style": ("looks great","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Tone focused ──────────────────────────────────────────────────────────
    ("I am absolutely livid, these are falling apart already",
     _gt({"durability": ("wears out fast","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("a bit let down, expected better quality for this brand",
     _gt({"intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("wondering if the suede version is more comfortable?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("delivered on time, standard packaging, no issues",
     _gt({"intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("these are genuinely the best shoes I have ever bought",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Mixed sentiment ───────────────────────────────────────────────────────
    ("the comfort is incredible but the durability is terrible",
     _gt({"comfort": ("very comfortable","classified"), "durability": ("wears out fast","classified"),
          "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("stylish and fits well but way too expensive for students",
     _gt({"style": ("looks great","classified"), "fit": ("true to size","classified"),
          "price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("comfortable but already showing wear after just three weeks",
     _gt({"comfort": ("very comfortable","classified"), "durability": ("wears out fast","classified"),
          "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Neutral informational ─────────────────────────────────────────────────
    ("arrived in good condition, matches the product description",
     _gt({"intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("standard shoes, they do the job without anything special",
     _gt({"intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("used them for two weeks, nothing remarkable to report",
     _gt({"intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    # ── Final varied batch ────────────────────────────────────────────────────
    ("I genuinely cannot go back to my old shoes after wearing these",
     _gt({"comfort": ("very comfortable","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("the stitching is already unraveling, really annoying",
     _gt({"durability": ("wears out fast","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("still look and feel great after six months of daily wear",
     _gt({"durability": ("lasts well","classified"), "style": ("looks great","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("this is my third pair, says everything about the quality",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("the price went up but they're still worth buying",
     _gt({"price_value": ("worth it","classified"), "intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    ("tight, ugly, overpriced — the trifecta of disappointment",
     _gt({"fit": ("too tight","classified"), "style": ("looks cheap","classified"),
          "price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("no strong feelings either way, they're just shoes",
     _gt({"intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("hard to judge after only two wears, seem decent",
     _gt({"durability": ("too early to tell","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),
]
