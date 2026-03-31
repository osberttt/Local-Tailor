"""
shops/womens_clothing/synthetic.py
===================================
Synthetic women's clothing shop dataset with labeled comments.

Dimensions:
  - fit_sizing
  - material_quality
  - style_appearance
  - comfort_wearability
"""

from __future__ import annotations
from typing import Dict, List


# ── Dimensions for this shop ────────────────────────────────────────────────

ALL_DIMS = [
    "fit_sizing",
    "material_quality",
    "style_appearance",
    "comfort_wearability"
]


# ── Helpers ────────────────────────────────────────────────────────────────

def _na(dims: List[str]) -> Dict:
    return {d: {"value": "N/A", "flag": "na"} for d in dims}


def _gt(overrides: Dict) -> Dict:
    """Build ground truth: start with N/A for all dims, apply overrides."""
    gt = _na(ALL_DIMS)
    for dim, value in overrides.items():
        gt[dim] = {"value": value, "flag": "classified"}
    return gt


# ── Dataset ────────────────────────────────────────────────────────────────

COMMENTS_RAW = [

    # ── Fit & Sizing ─────────────────────────────────────────────────────────
    ("ordered my usual size and it was way too tight",
     _gt({"fit_sizing": "runs small"})),

    ("fits perfectly in my usual size",
     _gt({"fit_sizing": "true to size"})),

    ("way too big, could have sized down",
     _gt({"fit_sizing": "runs large"})),

    ("tight across the chest, need to size up",
     _gt({"fit_sizing": "runs small"})),

    ("loose and oversized even in my normal size",
     _gt({"fit_sizing": "runs large"})),

    ("fits exactly as expected, no sizing issues",
     _gt({"fit_sizing": "true to size"})),

    # ── Material Quality ─────────────────────────────────────────────────────
    ("fabric feels cheap and thin",
     _gt({"material_quality": "low quality"})),

    ("excellent construction and premium fabric",
     _gt({"material_quality": "high quality"})),

    ("seams came apart after one wash",
     _gt({"material_quality": "low quality"})),

    ("just received, too early to judge quality",
     _gt({"material_quality": "uncertain"})),

    ("material feels luxurious and well made",
     _gt({"material_quality": "high quality"})),

    ("fabric pills after a few wears",
     _gt({"material_quality": "low quality"})),

    # ── Style & Appearance ───────────────────────────────────────────────────
    ("so flattering, makes me look amazing",
     _gt({"style_appearance": "flattering"})),

    ("not flattering at all, looks boxy",
     _gt({"style_appearance": "unflattering"})),

    ("looks nothing like the photo",
     _gt({"style_appearance": "different from expectations"})),

    ("the color is different in person",
     _gt({"style_appearance": "different from expectations"})),

    ("very flattering silhouette",
     _gt({"style_appearance": "flattering"})),

    ("cut is unflattering for my body type",
     _gt({"style_appearance": "unflattering"})),

    # ── Comfort ──────────────────────────────────────────────────────────────
    ("so comfortable, could wear all day",
     _gt({"comfort_wearability": "very comfortable"})),

    ("average comfort, nothing special",
     _gt({"comfort_wearability": "moderately comfortable"})),

    ("itchy and uncomfortable fabric",
     _gt({"comfort_wearability": "uncomfortable"})),

    ("feels amazing and soft",
     _gt({"comfort_wearability": "very comfortable"})),

    ("ok to wear but not very comfortable",
     _gt({"comfort_wearability": "moderately comfortable"})),

    ("too tight and restrictive to move",
     _gt({"comfort_wearability": "uncomfortable"})),

    # ── Multi-dimension examples ─────────────────────────────────────────────
    ("flattering fit but runs very small",
     _gt({
         "fit_sizing": "runs small",
         "style_appearance": "flattering"
     })),

    ("great quality but uncomfortable to wear",
     _gt({
         "material_quality": "high quality",
         "comfort_wearability": "uncomfortable"
     })),

    ("fits perfectly and very comfortable",
     _gt({
         "fit_sizing": "true to size",
         "comfort_wearability": "very comfortable"
     })),

    ("cheap material and runs large",
     _gt({
         "material_quality": "low quality",
         "fit_sizing": "runs large"
     })),

    ("beautiful style but material feels cheap",
     _gt({
         "style_appearance": "flattering",
         "material_quality": "low quality"
     })),

    ("perfect fit and great quality overall",
     _gt({
         "fit_sizing": "true to size",
         "material_quality": "high quality"
     })),

]