"""
shops/pillow/synthetic.py
=========================
Pillow evaluation dataset — 212 comments (112 + 100).

Part 1 (rows 1–112): the 8 training examples for each of the 14 classes,
  copied verbatim from examples.json. Single-dimension labels.

Part 2 (rows 113–212): ~100 combo rows built by taking the first clause of
  one example and the second clause of another (different dimension).
  Two-dimension labels. Language mirrors examples directly.
"""

from __future__ import annotations
from typing import Dict, List

ALL_DIMS = ["comfort", "shape", "durability", "price_value"]

def _na(dims: List[str]) -> Dict:
    return {d: {"value": "N/A", "flag": "na"} for d in dims}

def _gt(overrides: Dict) -> Dict:
    gt = _na(ALL_DIMS)
    for dim, (value, flag) in overrides.items():
        gt[dim] = {"value": value, "flag": flag}
    return gt


COMMENTS_RAW = [

    # ══════════════════════════════════════════════════════════════════════════
    # PART 1 — Examples verbatim (single-dim, 8 per class)
    # ══════════════════════════════════════════════════════════════════════════

    # ── comfort: too firm ─────────────────────────────────────────────────────
    ("A bit too firm for me, not comfortable for side sleeping",
     _gt({"comfort": ("too firm","classified")})),
    ("The pillow is on the firm side, I would prefer something softer",
     _gt({"comfort": ("too firm","classified")})),
    ("Firmer than I expected, my neck feels stiff in the mornings",
     _gt({"comfort": ("too firm","classified")})),
    ("A touch too hard for a comfortable night's sleep",
     _gt({"comfort": ("too firm","classified")})),
    ("The filling is quite dense and not soft enough for my taste",
     _gt({"comfort": ("too firm","classified")})),
    ("Too firm for me, I find it difficult to get comfortable on it",
     _gt({"comfort": ("too firm","classified")})),
    ("Harder than the description suggested, not what I was looking for",
     _gt({"comfort": ("too firm","classified")})),
    ("The pillow is quite stiff, would suit someone who prefers a hard feel",
     _gt({"comfort": ("too firm","classified")})),

    # ── comfort: just right ───────────────────────────────────────────────────
    ("The firmness is just right, very comfortable for sleeping",
     _gt({"comfort": ("just right","classified")})),
    ("Perfect balance of soft and supportive, really pleased with the comfort",
     _gt({"comfort": ("just right","classified")})),
    ("Comfortable from the first night, the feel is exactly what I needed",
     _gt({"comfort": ("just right","classified")})),
    ("The comfort level is spot on, not too firm and not too soft",
     _gt({"comfort": ("just right","classified")})),
    ("Soft enough to be comfortable but firm enough to support my neck well",
     _gt({"comfort": ("just right","classified")})),
    ("Really comfortable pillow, woke up without any stiffness",
     _gt({"comfort": ("just right","classified")})),
    ("The softness is ideal, suits my sleeping position perfectly",
     _gt({"comfort": ("just right","classified")})),
    ("Great comfort level, very happy with how it feels",
     _gt({"comfort": ("just right","classified")})),

    # ── comfort: too soft ─────────────────────────────────────────────────────
    ("A bit too soft for me, lacks the support I need for my neck",
     _gt({"comfort": ("too soft","classified")})),
    ("The pillow compresses too much under my head during the night",
     _gt({"comfort": ("too soft","classified")})),
    ("Too soft to provide proper neck support",
     _gt({"comfort": ("too soft","classified")})),
    ("Feels pleasant but too soft for anyone needing firmer support",
     _gt({"comfort": ("too soft","classified")})),
    ("The filling is quite soft and my head sinks in too far",
     _gt({"comfort": ("too soft","classified")})),
    ("Soft and comfortable but lacks the support I was hoping for",
     _gt({"comfort": ("too soft","classified")})),
    ("A little too soft for my liking, I find myself readjusting it through the night",
     _gt({"comfort": ("too soft","classified")})),
    ("Too soft for back sleeping, not enough resistance under my neck",
     _gt({"comfort": ("too soft","classified")})),

    # ── comfort: changes over time ────────────────────────────────────────────
    ("Was comfortable when new but has become firmer over the past few months",
     _gt({"comfort": ("changes over time","classified")})),
    ("The feel has changed quite noticeably since I first started using it",
     _gt({"comfort": ("changes over time","classified")})),
    ("Good comfort for the first few weeks but it has declined since then",
     _gt({"comfort": ("changes over time","classified")})),
    ("Noticed the pillow becoming harder after a few months of regular use",
     _gt({"comfort": ("changes over time","classified")})),
    ("Started off nicely soft but the filling has compacted over time",
     _gt({"comfort": ("changes over time","classified")})),
    ("Was a decent pillow at first but the comfort level has dropped",
     _gt({"comfort": ("changes over time","classified")})),
    ("The softness faded after a few washes, not as comfortable as when new",
     _gt({"comfort": ("changes over time","classified")})),
    ("Comfortable initially but gradually became less so after a couple of months",
     _gt({"comfort": ("changes over time","classified")})),

    # ── shape: too thin ───────────────────────────────────────────────────────
    ("The pillow is quite flat, not enough height for side sleeping",
     _gt({"shape": ("too thin","classified")})),
    ("Thinner than I expected, not enough loft for proper neck support",
     _gt({"shape": ("too thin","classified")})),
    ("Not enough height for me, feels quite flat under my head",
     _gt({"shape": ("too thin","classified")})),
    ("A little thin compared to what I was hoping for",
     _gt({"shape": ("too thin","classified")})),
    ("The loft is on the low side, I need a thicker pillow",
     _gt({"shape": ("too thin","classified")})),
    ("Too flat for side sleeping, needs more thickness",
     _gt({"shape": ("too thin","classified")})),
    ("Not as thick as it appeared in the product photos",
     _gt({"shape": ("too thin","classified")})),
    ("Quite flat overall, which makes it unsuitable for anyone needing more neck support",
     _gt({"shape": ("too thin","classified")})),

    # ── shape: just right thickness ───────────────────────────────────────────
    ("The height is just right for me as a side sleeper",
     _gt({"shape": ("just right thickness","classified")})),
    ("Perfect loft, keeps my neck nicely aligned through the night",
     _gt({"shape": ("just right thickness","classified")})),
    ("The thickness is exactly what I was looking for",
     _gt({"shape": ("just right thickness","classified")})),
    ("Just the right height, works well for back and side sleeping",
     _gt({"shape": ("just right thickness","classified")})),
    ("Great loft, not too flat and not too high",
     _gt({"shape": ("just right thickness","classified")})),
    ("The pillow height suits me very well",
     _gt({"shape": ("just right thickness","classified")})),
    ("Good thickness overall, well suited to my sleeping style",
     _gt({"shape": ("just right thickness","classified")})),
    ("The loft is just right, comfortable and supportive",
     _gt({"shape": ("just right thickness","classified")})),

    # ── shape: too thick ──────────────────────────────────────────────────────
    ("A bit too tall for me, pushes my neck up at an awkward angle",
     _gt({"shape": ("too thick","classified")})),
    ("The pillow is quite thick, which does not suit stomach sleepers",
     _gt({"shape": ("too thick","classified")})),
    ("Too high for my preference, causes some neck discomfort",
     _gt({"shape": ("too thick","classified")})),
    ("A little too puffy, the extra height does not work for me",
     _gt({"shape": ("too thick","classified")})),
    ("The loft is higher than I prefer and causes some strain",
     _gt({"shape": ("too thick","classified")})),
    ("Too tall for my needs, I would prefer a flatter option",
     _gt({"shape": ("too thick","classified")})),
    ("The height is a bit much for how I sleep",
     _gt({"shape": ("too thick","classified")})),
    ("Quite a high pillow, not ideal for those who prefer a lower profile",
     _gt({"shape": ("too thick","classified")})),

    # ── shape: loses shape ────────────────────────────────────────────────────
    ("The pillow has started to go flat after a couple of months of use",
     _gt({"shape": ("loses shape","classified")})),
    ("Loses its shape fairly quickly, does not maintain the original loft",
     _gt({"shape": ("loses shape","classified")})),
    ("The filling tends to shift and clump after regular use",
     _gt({"shape": ("loses shape","classified")})),
    ("Does not hold its shape as well as I had hoped",
     _gt({"shape": ("loses shape","classified")})),
    ("Has gone a bit flat since I started using it regularly",
     _gt({"shape": ("loses shape","classified")})),
    ("The pillow bunches to one side and becomes uneven",
     _gt({"shape": ("loses shape","classified")})),
    ("Lost its original shape after a few months",
     _gt({"shape": ("loses shape","classified")})),
    ("The filling shifts around and the pillow no longer keeps a consistent shape",
     _gt({"shape": ("loses shape","classified")})),

    # ── durability: lasts well ────────────────────────────────────────────────
    ("Still in great condition after over a year of daily use",
     _gt({"durability": ("lasts well","classified")})),
    ("Holds up very well over time, no signs of deterioration",
     _gt({"durability": ("lasts well","classified")})),
    ("Very durable, still performs as well as it did when new",
     _gt({"durability": ("lasts well","classified")})),
    ("Has lasted well through regular washing and daily use",
     _gt({"durability": ("lasts well","classified")})),
    ("No noticeable signs of wear after several months, good quality",
     _gt({"durability": ("lasts well","classified")})),
    ("Washes well and maintains its shape, very durable",
     _gt({"durability": ("lasts well","classified")})),
    ("Durable construction, still in excellent condition after regular use",
     _gt({"durability": ("lasts well","classified")})),
    ("Good longevity, the pillow has held up better than I expected",
     _gt({"durability": ("lasts well","classified")})),

    # ── durability: degrades quickly ──────────────────────────────────────────
    ("Started to flatten out sooner than I would have expected",
     _gt({"durability": ("degrades quickly","classified")})),
    ("Shows signs of wear after only a couple of months of regular use",
     _gt({"durability": ("degrades quickly","classified")})),
    ("The filling has compacted much faster than it should",
     _gt({"durability": ("degrades quickly","classified")})),
    ("Not as durable as I hoped, wearing out earlier than expected",
     _gt({"durability": ("degrades quickly","classified")})),
    ("Has worn out more quickly than expected for the price paid",
     _gt({"durability": ("degrades quickly","classified")})),
    ("The cover started to look worn after just a few months",
     _gt({"durability": ("degrades quickly","classified")})),
    ("Deteriorated quicker than I expected, a bit disappointing",
     _gt({"durability": ("degrades quickly","classified")})),
    ("The quality dropped off faster than it should have",
     _gt({"durability": ("degrades quickly","classified")})),

    # ── durability: too early to tell ─────────────────────────────────────────
    ("Just received it, too early to say how long it will last",
     _gt({"durability": ("too early to tell","classified")})),
    ("Only had it for a few days, need more time to assess",
     _gt({"durability": ("too early to tell","classified")})),
    ("Brand new, will update this review after a few weeks of use",
     _gt({"durability": ("too early to tell","classified")})),
    ("Only used it a handful of times so far",
     _gt({"durability": ("too early to tell","classified")})),
    ("First impressions are good but it is still too early to judge",
     _gt({"durability": ("too early to tell","classified")})),
    ("New purchase, cannot comment on longevity at this stage",
     _gt({"durability": ("too early to tell","classified")})),
    ("Too soon to tell, but it seems decent quality so far",
     _gt({"durability": ("too early to tell","classified")})),
    ("Just started using it, will have a better sense of durability in time",
     _gt({"durability": ("too early to tell","classified")})),

    # ── price_value: too expensive ────────────────────────────────────────────
    ("A bit overpriced for the quality you receive",
     _gt({"price_value": ("too expensive","classified")})),
    ("The price feels high for what is essentially a standard pillow",
     _gt({"price_value": ("too expensive","classified")})),
    ("Not great value for money compared to similar products",
     _gt({"price_value": ("too expensive","classified")})),
    ("A little expensive given what you actually get",
     _gt({"price_value": ("too expensive","classified")})),
    ("Could find comparable quality at a lower price elsewhere",
     _gt({"price_value": ("too expensive","classified")})),
    ("The quality does not quite match the asking price",
     _gt({"price_value": ("too expensive","classified")})),
    ("Priced a bit too high for what it delivers",
     _gt({"price_value": ("too expensive","classified")})),
    ("Not the best value, I have found better for less",
     _gt({"price_value": ("too expensive","classified")})),

    # ── price_value: good value ───────────────────────────────────────────────
    ("Reasonable price for the quality, happy with what I paid",
     _gt({"price_value": ("good value","classified")})),
    ("Good value for money overall",
     _gt({"price_value": ("good value","classified")})),
    ("Fair price given the quality received",
     _gt({"price_value": ("good value","classified")})),
    ("Solid value at this price point, no complaints",
     _gt({"price_value": ("good value","classified")})),
    ("The price is appropriate for the quality you get",
     _gt({"price_value": ("good value","classified")})),
    ("Good deal, decent quality for the money",
     _gt({"price_value": ("good value","classified")})),
    ("Reasonably priced and performs well",
     _gt({"price_value": ("good value","classified")})),
    ("Happy with the value, it is worth what I paid",
     _gt({"price_value": ("good value","classified")})),

    # ── price_value: worth it ─────────────────────────────────────────────────
    ("Worth every penny, the quality is excellent",
     _gt({"price_value": ("worth it","classified")})),
    ("A bit of a splurge but completely worth it",
     _gt({"price_value": ("worth it","classified")})),
    ("The higher price is justified by the quality you receive",
     _gt({"price_value": ("worth it","classified")})),
    ("Premium price but premium quality to match",
     _gt({"price_value": ("worth it","classified")})),
    ("Worth the investment, I would pay full price again",
     _gt({"price_value": ("worth it","classified")})),
    ("Yes it costs more but you get what you pay for",
     _gt({"price_value": ("worth it","classified")})),
    ("The quality makes it worth the asking price",
     _gt({"price_value": ("worth it","classified")})),
    ("Pricier than average but the quality makes it worthwhile",
     _gt({"price_value": ("worth it","classified")})),


    # ══════════════════════════════════════════════════════════════════════════
    # PART 2 — Shuffled combos (2-dim, first-half + second-half of examples)
    # ══════════════════════════════════════════════════════════════════════════

    # ── comfort × shape ───────────────────────────────────────────────────────
    ("A bit too firm for me, and the pillow is quite flat, not enough height for side sleeping",
     _gt({"comfort": ("too firm","classified"), "shape": ("too thin","classified")})),

    ("Firmer than I expected, and not enough loft for proper neck support",
     _gt({"comfort": ("too firm","classified"), "shape": ("too thin","classified")})),

    ("The pillow is on the firm side, though the height is just right for me as a side sleeper",
     _gt({"comfort": ("too firm","classified"), "shape": ("just right thickness","classified")})),

    ("The filling is quite dense and not soft enough for my taste, though the pillow height suits me very well",
     _gt({"comfort": ("too firm","classified"), "shape": ("just right thickness","classified")})),

    ("A bit too firm for me and a little too puffy, the extra height does not work for me",
     _gt({"comfort": ("too firm","classified"), "shape": ("too thick","classified")})),

    ("The pillow is quite stiff and too high for my preference, causes some neck discomfort",
     _gt({"comfort": ("too firm","classified"), "shape": ("too thick","classified")})),

    ("Harder than the description suggested, and the filling tends to shift and clump after regular use",
     _gt({"comfort": ("too firm","classified"), "shape": ("loses shape","classified")})),

    ("Too firm for me, I find it difficult to get comfortable on it, and the pillow bunches to one side and becomes uneven",
     _gt({"comfort": ("too firm","classified"), "shape": ("loses shape","classified")})),

    ("The comfort level is spot on, though the pillow is quite flat, not enough height for side sleeping",
     _gt({"comfort": ("just right","classified"), "shape": ("too thin","classified")})),

    ("Really comfortable pillow, woke up without any stiffness, but not enough loft for proper neck support",
     _gt({"comfort": ("just right","classified"), "shape": ("too thin","classified")})),

    ("The firmness is just right and the height is just right for me as a side sleeper",
     _gt({"comfort": ("just right","classified"), "shape": ("just right thickness","classified")})),

    ("Perfect balance of soft and supportive, and the loft is just right, comfortable and supportive",
     _gt({"comfort": ("just right","classified"), "shape": ("just right thickness","classified")})),

    ("Great comfort level, very happy with how it feels, but a bit too tall, pushes my neck up at an awkward angle",
     _gt({"comfort": ("just right","classified"), "shape": ("too thick","classified")})),

    ("Comfortable from the first night, the feel is exactly what I needed, but the filling tends to shift and clump after regular use",
     _gt({"comfort": ("just right","classified"), "shape": ("loses shape","classified")})),

    ("The softness is ideal, suits my sleeping position perfectly, but the pillow has started to go flat after a couple of months of use",
     _gt({"comfort": ("just right","classified"), "shape": ("loses shape","classified")})),

    ("A bit too soft for me, lacks the support I need, and the pillow is quite flat, not enough height for side sleeping",
     _gt({"comfort": ("too soft","classified"), "shape": ("too thin","classified")})),

    ("The filling is quite soft and my head sinks in too far, and not enough loft for proper neck support",
     _gt({"comfort": ("too soft","classified"), "shape": ("too thin","classified")})),

    ("Too soft to provide proper neck support, though the thickness is exactly what I was looking for",
     _gt({"comfort": ("too soft","classified"), "shape": ("just right thickness","classified")})),

    ("A little too soft for my liking, and also quite thick, the loft is higher than I prefer",
     _gt({"comfort": ("too soft","classified"), "shape": ("too thick","classified")})),

    ("Soft and comfortable but lacks the support I was hoping for, and the pillow has started to go flat after a couple of months of use",
     _gt({"comfort": ("too soft","classified"), "shape": ("loses shape","classified")})),

    ("The softness faded after a few washes, not as comfortable as when new, and the pillow has also lost its original shape",
     _gt({"comfort": ("changes over time","classified"), "shape": ("loses shape","classified")})),

    ("Good comfort for the first few weeks but it has declined since then, and the filling shifts around and the pillow no longer keeps a consistent shape",
     _gt({"comfort": ("changes over time","classified"), "shape": ("loses shape","classified")})),

    ("The feel has changed quite noticeably since I first started using it, though the thickness is exactly what I was looking for",
     _gt({"comfort": ("changes over time","classified"), "shape": ("just right thickness","classified")})),

    ("Was comfortable when new but has become firmer, and the pillow is quite flat with not enough height",
     _gt({"comfort": ("changes over time","classified"), "shape": ("too thin","classified")})),

    ("Noticed the pillow becoming harder after a few months of regular use, and now a bit too tall, pushes my neck up at an awkward angle",
     _gt({"comfort": ("changes over time","classified"), "shape": ("too thick","classified")})),

    # ── comfort × price_value ─────────────────────────────────────────────────
    ("A bit too firm for me, and the price feels high for what is essentially a standard pillow",
     _gt({"comfort": ("too firm","classified"), "price_value": ("too expensive","classified")})),

    ("The pillow is quite stiff and a bit overpriced for the quality you receive",
     _gt({"comfort": ("too firm","classified"), "price_value": ("too expensive","classified")})),

    ("Too firm for me, I find it difficult to get comfortable on it, but the price is appropriate for the quality you get",
     _gt({"comfort": ("too firm","classified"), "price_value": ("good value","classified")})),

    ("The filling is quite dense and not soft enough for my taste, but the quality makes it worth the asking price",
     _gt({"comfort": ("too firm","classified"), "price_value": ("worth it","classified")})),

    ("The firmness is just right, though a bit overpriced for the quality you receive",
     _gt({"comfort": ("just right","classified"), "price_value": ("too expensive","classified")})),

    ("Really comfortable pillow, woke up without any stiffness, and reasonable price for the quality, happy with what I paid",
     _gt({"comfort": ("just right","classified"), "price_value": ("good value","classified")})),

    ("The comfort level is spot on, not too firm and not too soft, and solid value at this price point, no complaints",
     _gt({"comfort": ("just right","classified"), "price_value": ("good value","classified")})),

    ("Great comfort level, very happy with how it feels, and worth every penny, the quality is excellent",
     _gt({"comfort": ("just right","classified"), "price_value": ("worth it","classified")})),

    ("The firmness is just right, very comfortable for sleeping, and the higher price is justified by the quality you receive",
     _gt({"comfort": ("just right","classified"), "price_value": ("worth it","classified")})),

    ("Too soft to provide proper neck support and not great value for money compared to similar products",
     _gt({"comfort": ("too soft","classified"), "price_value": ("too expensive","classified")})),

    ("A bit too soft for me, lacks the support I need, but fair price given the quality received",
     _gt({"comfort": ("too soft","classified"), "price_value": ("good value","classified")})),

    ("Was comfortable when new but has become firmer, and the price feels high for what is essentially a standard pillow",
     _gt({"comfort": ("changes over time","classified"), "price_value": ("too expensive","classified")})),

    ("Good comfort for the first few weeks but it has declined since then, and not great value for money compared to similar products",
     _gt({"comfort": ("changes over time","classified"), "price_value": ("too expensive","classified")})),

    ("The feel has changed quite noticeably since I first started using it, though the price is fair given the quality received",
     _gt({"comfort": ("changes over time","classified"), "price_value": ("good value","classified")})),

    ("Started off nicely soft but the filling has compacted over time, though it was worth the investment when it was new",
     _gt({"comfort": ("changes over time","classified"), "price_value": ("worth it","classified")})),

    # ── comfort × durability ──────────────────────────────────────────────────
    ("The filling is quite dense and not soft enough for my taste, but it holds up very well over time, no signs of deterioration",
     _gt({"comfort": ("too firm","classified"), "durability": ("lasts well","classified")})),

    ("Firmer than I expected, my neck feels stiff in the mornings, and shows signs of wear after only a couple of months",
     _gt({"comfort": ("too firm","classified"), "durability": ("degrades quickly","classified")})),

    ("Soft enough to be comfortable but firm enough to support my neck well, and it holds up very well over time, no signs of deterioration",
     _gt({"comfort": ("just right","classified"), "durability": ("lasts well","classified")})),

    ("The firmness is just right, very comfortable for sleeping, and very durable, still performs as well as it did when new",
     _gt({"comfort": ("just right","classified"), "durability": ("lasts well","classified")})),

    ("Comfortable from the first night, the feel is exactly what I needed, but the filling has compacted much faster than it should",
     _gt({"comfort": ("just right","classified"), "durability": ("degrades quickly","classified")})),

    ("Great comfort level, very happy with how it feels, just received it and too early to say how long it will last",
     _gt({"comfort": ("just right","classified"), "durability": ("too early to tell","classified")})),

    ("Too soft for back sleeping, not enough resistance under my neck, but it holds up very well over time, no signs of deterioration",
     _gt({"comfort": ("too soft","classified"), "durability": ("lasts well","classified")})),

    ("The pillow compresses too much under my head and shows signs of wear after only a couple of months of regular use",
     _gt({"comfort": ("too soft","classified"), "durability": ("degrades quickly","classified")})),

    ("A bit too soft for me, lacks the support I need, just received it and too early to say how long it will last",
     _gt({"comfort": ("too soft","classified"), "durability": ("too early to tell","classified")})),

    ("The softness faded after a few washes, not as comfortable as when new, and the quality dropped off faster than it should have",
     _gt({"comfort": ("changes over time","classified"), "durability": ("degrades quickly","classified")})),

    ("Started off nicely soft but the filling has compacted over time, deteriorating quicker than I expected",
     _gt({"comfort": ("changes over time","classified"), "durability": ("degrades quickly","classified")})),

    ("Was a decent pillow at first but the comfort level has dropped, though it has lasted well through regular washing and daily use",
     _gt({"comfort": ("changes over time","classified"), "durability": ("lasts well","classified")})),

    # ── shape × price_value ───────────────────────────────────────────────────
    ("Quite flat overall, which makes it unsuitable for anyone needing more neck support, and a bit overpriced for the quality you receive",
     _gt({"shape": ("too thin","classified"), "price_value": ("too expensive","classified")})),

    ("Not enough height for me, feels quite flat under my head, and not the best value, I have found better for less",
     _gt({"shape": ("too thin","classified"), "price_value": ("too expensive","classified")})),

    ("The loft is on the low side, I need a thicker pillow, but fair price given the quality received",
     _gt({"shape": ("too thin","classified"), "price_value": ("good value","classified")})),

    ("The height is just right for me as a side sleeper, though the price feels high for what is essentially a standard pillow",
     _gt({"shape": ("just right thickness","classified"), "price_value": ("too expensive","classified")})),

    ("The thickness is exactly what I was looking for, and good value for money overall",
     _gt({"shape": ("just right thickness","classified"), "price_value": ("good value","classified")})),

    ("Perfect loft, keeps my neck nicely aligned through the night, and worth every penny, the quality is excellent",
     _gt({"shape": ("just right thickness","classified"), "price_value": ("worth it","classified")})),

    ("Good thickness overall, well suited to my sleeping style, and worth the investment, I would pay full price again",
     _gt({"shape": ("just right thickness","classified"), "price_value": ("worth it","classified")})),

    ("The height is a bit much for how I sleep, and the price feels high for what is essentially a standard pillow",
     _gt({"shape": ("too thick","classified"), "price_value": ("too expensive","classified")})),

    ("Too high for my preference, causes some neck discomfort, but reasonably priced and performs well",
     _gt({"shape": ("too thick","classified"), "price_value": ("good value","classified")})),

    ("The filling tends to shift and clump after regular use, and not great value for money compared to similar products",
     _gt({"shape": ("loses shape","classified"), "price_value": ("too expensive","classified")})),

    ("Does not hold its shape as well as I had hoped, but the price is appropriate for the quality you get",
     _gt({"shape": ("loses shape","classified"), "price_value": ("good value","classified")})),

    ("The pillow bunches to one side and becomes uneven, but the quality makes it worth the asking price",
     _gt({"shape": ("loses shape","classified"), "price_value": ("worth it","classified")})),

    # ── shape × durability ────────────────────────────────────────────────────
    ("Not enough height for me, feels quite flat under my head, but it holds up very well over time, no signs of deterioration",
     _gt({"shape": ("too thin","classified"), "durability": ("lasts well","classified")})),

    ("The pillow is quite flat, not enough height for side sleeping, and has worn out more quickly than expected for the price paid",
     _gt({"shape": ("too thin","classified"), "durability": ("degrades quickly","classified")})),

    ("The height is just right for me as a side sleeper, and it holds up very well over time, no signs of deterioration",
     _gt({"shape": ("just right thickness","classified"), "durability": ("lasts well","classified")})),

    ("Great loft, not too flat and not too high, and very durable, still performs as well as it did when new",
     _gt({"shape": ("just right thickness","classified"), "durability": ("lasts well","classified")})),

    ("Good thickness overall, well suited to my sleeping style, but the quality dropped off faster than it should have",
     _gt({"shape": ("just right thickness","classified"), "durability": ("degrades quickly","classified")})),

    ("The thickness is exactly what I was looking for, just received it and too early to say how long it will last",
     _gt({"shape": ("just right thickness","classified"), "durability": ("too early to tell","classified")})),

    ("A bit too tall for me, pushes my neck up at an awkward angle, but very durable, still performs as well as it did when new",
     _gt({"shape": ("too thick","classified"), "durability": ("lasts well","classified")})),

    ("The loft is higher than I prefer and causes some strain, and the filling has compacted much faster than it should",
     _gt({"shape": ("too thick","classified"), "durability": ("degrades quickly","classified")})),

    ("Quite a high pillow, not ideal for those who prefer a lower profile, just received it and too early to say how long it will last",
     _gt({"shape": ("too thick","classified"), "durability": ("too early to tell","classified")})),

    ("The pillow has started to go flat after a couple of months of use, and the quality dropped off faster than it should have",
     _gt({"shape": ("loses shape","classified"), "durability": ("degrades quickly","classified")})),

    ("Lost its original shape after a few months, and shows signs of wear after only a couple of months of regular use",
     _gt({"shape": ("loses shape","classified"), "durability": ("degrades quickly","classified")})),

    ("The pillow bunches to one side and becomes uneven, but durable construction, still in excellent condition after regular use",
     _gt({"shape": ("loses shape","classified"), "durability": ("lasts well","classified")})),

    # ── durability × price_value ──────────────────────────────────────────────
    ("Holds up very well over time, no signs of deterioration, and good value for money overall",
     _gt({"durability": ("lasts well","classified"), "price_value": ("good value","classified")})),

    ("Durable construction, still in excellent condition after regular use, and reasonable price for the quality, happy with what I paid",
     _gt({"durability": ("lasts well","classified"), "price_value": ("good value","classified")})),

    ("Very durable, still performs as well as it did when new, and worth every penny, the quality is excellent",
     _gt({"durability": ("lasts well","classified"), "price_value": ("worth it","classified")})),

    ("Good longevity, the pillow has held up better than I expected, and the higher price is justified by the quality you receive",
     _gt({"durability": ("lasts well","classified"), "price_value": ("worth it","classified")})),

    ("Very durable, still performs as well as it did when new, but the price feels high for what is essentially a standard pillow",
     _gt({"durability": ("lasts well","classified"), "price_value": ("too expensive","classified")})),

    ("The filling has compacted much faster than it should, and not great value for money compared to similar products",
     _gt({"durability": ("degrades quickly","classified"), "price_value": ("too expensive","classified")})),

    ("Deteriorated quicker than I expected, a bit disappointing, and the price feels high for what is essentially a standard pillow",
     _gt({"durability": ("degrades quickly","classified"), "price_value": ("too expensive","classified")})),

    ("Shows signs of wear after only a couple of months of regular use, though the price is fair given the quality received",
     _gt({"durability": ("degrades quickly","classified"), "price_value": ("good value","classified")})),

    ("Has worn out more quickly than expected for the price paid, and not the best value, I have found better for less",
     _gt({"durability": ("degrades quickly","classified"), "price_value": ("too expensive","classified")})),

    ("Just received it, too early to say how long it will last, but the price seems reasonable for the quality",
     _gt({"durability": ("too early to tell","classified"), "price_value": ("good value","classified")})),

    ("Brand new, will update this review after a few weeks of use, solid value at this price point, no complaints",
     _gt({"durability": ("too early to tell","classified"), "price_value": ("good value","classified")})),

    ("Only used it a handful of times so far, but it feels like it could be worth the investment",
     _gt({"durability": ("too early to tell","classified"), "price_value": ("worth it","classified")})),

    ("Only had it for a few days, need more time to assess, but the price feels high for what is essentially a standard pillow",
     _gt({"durability": ("too early to tell","classified"), "price_value": ("too expensive","classified")})),

    # ── extra combos to reach 100 ─────────────────────────────────────────────
    ("A touch too hard for a comfortable night's sleep, and the pillow is quite flat, not enough height for side sleeping",
     _gt({"comfort": ("too firm","classified"), "shape": ("too thin","classified")})),

    ("The softness is ideal, suits my sleeping position perfectly, and great loft, not too flat and not too high",
     _gt({"comfort": ("just right","classified"), "shape": ("just right thickness","classified")})),

    ("A little too soft for my liking, and too high for my preference, causes some neck discomfort",
     _gt({"comfort": ("too soft","classified"), "shape": ("too thick","classified")})),

    ("Comfortable initially but gradually became less so after a couple of months, and the filling shifts around and the pillow no longer keeps a consistent shape",
     _gt({"comfort": ("changes over time","classified"), "shape": ("loses shape","classified")})),

    ("Harder than the description suggested, not what I was looking for, just received it and too early to say how long it will last",
     _gt({"comfort": ("too firm","classified"), "durability": ("too early to tell","classified")})),

    ("Too soft for back sleeping, not enough resistance under my neck, and not as durable as I hoped, wearing out earlier than expected",
     _gt({"comfort": ("too soft","classified"), "durability": ("degrades quickly","classified")})),

    ("Thinner than I expected, not enough loft for proper neck support, just received it and too early to say how long it will last",
     _gt({"shape": ("too thin","classified"), "durability": ("too early to tell","classified")})),

    ("Loses its shape fairly quickly, does not maintain the original loft, just received it and too early to say how long it will last",
     _gt({"shape": ("loses shape","classified"), "durability": ("too early to tell","classified")})),

    ("A bit too tall for me, pushes my neck up at an awkward angle, and not great value for money compared to similar products",
     _gt({"shape": ("too thick","classified"), "price_value": ("too expensive","classified")})),

    ("Just started using it, will have a better sense of durability in time, but the price seems reasonable for the quality",
     _gt({"durability": ("too early to tell","classified"), "price_value": ("good value","classified")})),

    ("Washes well and maintains its shape, very durable, and pricier than average but the quality makes it worthwhile",
     _gt({"durability": ("lasts well","classified"), "price_value": ("worth it","classified")})),
]
