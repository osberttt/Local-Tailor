"""
localtailor/synthetic.py
========================
Generates a 150-comment synthetic pillow shop dataset with ground truth labels.

Comments are designed to test realistic edge cases:
  - Multi-dimension comments (one comment covers comfort + price + shape)
  - Short/ambiguous comments
  - Sarcastic comments
  - Questions (intent: needs reply)
  - N/A dimensions (comment mentions one thing, silent on others)

Ground truth schema per comment:
  {
    "comment_id": "c001",
    "message": "...",
    "ground_truth": {
      "comfort":     {"value": "too soft", "flag": "classified"},
      "shape":       {"value": "N/A",      "flag": "na"},
      "durability":  {"value": "N/A",      "flag": "na"},
      "price_value": {"value": "N/A",      "flag": "na"},
      "intent":      {"value": "negative review", "flag": "classified"},
      "tone":        {"value": "disappointed", "flag": "classified"}
    }
  }
"""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List


# ── Comment definitions ───────────────────────────────────────────────────────
# Each entry: (message, ground_truth_dict)
# ground_truth keys match dimension names in dimensions.yaml

def _na(dims: List[str]) -> Dict:
    return {d: {"value": "N/A", "flag": "na"} for d in dims}

ALL_DIMS = ["comfort", "shape", "durability", "price_value", "intent", "tone"]

def _gt(overrides: Dict) -> Dict:
    """Build ground truth: start with N/A for all dims, apply overrides."""
    gt = _na(ALL_DIMS)
    for dim, (value, flag) in overrides.items():
        gt[dim] = {"value": value, "flag": flag}
    return gt


COMMENTS_RAW = [
    # ── Comfort: too firm ─────────────────────────────────────────────────────
    ("this pillow is rock hard, my neck is in agony every morning",
     _gt({"comfort": ("too firm","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("very firm pillow, not comfortable for me but my husband loves it",
     _gt({"comfort": ("too firm","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("way too stiff, I put a blanket under my head instead now",
     _gt({"comfort": ("too firm","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("hard as a rock, nothing like the soft feel described",
     _gt({"comfort": ("too firm","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("not comfortable at all, way too rigid for side sleeping",
     _gt({"comfort": ("too firm","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Comfort: just right ───────────────────────────────────────────────────
    ("perfect firmness, woke up without neck pain for the first time in years",
     _gt({"comfort": ("just right","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("the softness is exactly right, not too firm not too soft",
     _gt({"comfort": ("just right","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("comfort is spot on, best sleep I've had in a while",
     _gt({"comfort": ("just right","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("ideal support level, my chiropractor would approve",
     _gt({"comfort": ("just right","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("feels great, the firmness is balanced really well",
     _gt({"comfort": ("just right","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Comfort: too soft ─────────────────────────────────────────────────────
    ("absolutely no support, my head sinks right through to the mattress",
     _gt({"comfort": ("too soft","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("too soft for anyone who needs actual neck support",
     _gt({"comfort": ("too soft","classified"), "intent": ("negative review","classified"), "tone": ("neutral","classified")})),

    ("squishy to the point of being useless, zero firmness",
     _gt({"comfort": ("too soft","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("like sleeping on a cloud but in a bad way, no support whatsoever",
     _gt({"comfort": ("too soft","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("way too mushy, I needed to fold it in half just to get some height",
     _gt({"comfort": ("too soft","classified"), "shape": ("too thin","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Comfort: changes over time ────────────────────────────────────────────
    ("was amazing for the first month but has gone really hard since then",
     _gt({"comfort": ("changes over time","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("comfort completely changed after the third wash, much stiffer now",
     _gt({"comfort": ("changes over time","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("started perfect but the feel shifted after a few weeks of use",
     _gt({"comfort": ("changes over time","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Shape: too thin ───────────────────────────────────────────────────────
    ("barely any loft to this pillow, way too flat for me",
     _gt({"shape": ("too thin","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("thinner than it looked online, not enough height for my needs",
     _gt({"shape": ("too thin","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("completely flat, not suitable for side sleepers at all",
     _gt({"shape": ("too thin","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("the pillow has no loft, my neck floats above it basically",
     _gt({"shape": ("too thin","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Shape: just right ─────────────────────────────────────────────────────
    ("the height is perfect for me as a back sleeper",
     _gt({"shape": ("just right thickness","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("perfect loft, keeps my spine perfectly aligned all night",
     _gt({"shape": ("just right thickness","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("great thickness, not too high not too low",
     _gt({"shape": ("just right thickness","classified"), "intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    # ── Shape: too thick ──────────────────────────────────────────────────────
    ("way too tall, my neck is strained from the angle it puts me at",
     _gt({"shape": ("too thick","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("too high for me, I'm a stomach sleeper and this just doesn't work",
     _gt({"shape": ("too thick","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("too puffy, great for someone who likes height but not for me",
     _gt({"shape": ("too thick","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    # ── Shape: loses shape ────────────────────────────────────────────────────
    ("went completely flat within two weeks, totally lost its form",
     _gt({"shape": ("loses shape","classified"), "durability": ("degrades quickly","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("doesn't hold its shape at all, collapses under the slightest weight",
     _gt({"shape": ("loses shape","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("the filling keeps bunching to one side, impossible to fix",
     _gt({"shape": ("loses shape","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("looked great out of the box but lost its shape within a month",
     _gt({"shape": ("loses shape","classified"), "durability": ("degrades quickly","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Durability: lasts well ────────────────────────────────────────────────
    ("had this for two years and it still feels like new, incredible quality",
     _gt({"durability": ("lasts well","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("washes perfectly and keeps its shape, very durable",
     _gt({"durability": ("lasts well","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("no signs of wear after 14 months of daily use, impressive",
     _gt({"durability": ("lasts well","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("the quality is exceptional, holds up through everything",
     _gt({"durability": ("lasts well","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Durability: degrades quickly ──────────────────────────────────────────
    ("fell apart after three months, the stitching just gave up",
     _gt({"durability": ("degrades quickly","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("filling started leaking out within a few weeks of normal use",
     _gt({"durability": ("degrades quickly","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("terribly made, completely worn out in under four months",
     _gt({"durability": ("degrades quickly","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("the cover started pilling immediately, poor material quality",
     _gt({"durability": ("degrades quickly","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Durability: too early to tell ─────────────────────────────────────────
    ("just received it today, looks good so far but too early to say",
     _gt({"durability": ("too early to tell","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("only had it for two nights, first impressions are positive",
     _gt({"durability": ("too early to tell","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("brand new, will update review after a month of use",
     _gt({"durability": ("too early to tell","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    # ── Price: too expensive ──────────────────────────────────────────────────
    ("way overpriced for what you actually get, not worth it at all",
     _gt({"price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("I've had better pillows for a third of this price",
     _gt({"price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("premium price but definitely not premium quality",
     _gt({"price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("can't believe how much I paid for something this average",
     _gt({"price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    # ── Price: good value ─────────────────────────────────────────────────────
    ("decent quality for the price, no real complaints",
     _gt({"price_value": ("good value","classified"), "intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    ("reasonable price for what you get, happy with the purchase",
     _gt({"price_value": ("good value","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("the price is fair given the quality, would recommend",
     _gt({"price_value": ("good value","classified"), "intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    # ── Price: worth it ───────────────────────────────────────────────────────
    ("absolutely worth every penny, my best ever bedding purchase",
     _gt({"price_value": ("worth it","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("spent more than I planned but no regrets, the quality is outstanding",
     _gt({"price_value": ("worth it","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("yes it's expensive but you get exactly what you pay for",
     _gt({"price_value": ("worth it","classified"), "intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    # ── Multi-dimension comments ──────────────────────────────────────────────
    ("love the comfort but it went flat after a month and honestly overpriced",
     _gt({"comfort": ("just right","classified"), "shape": ("loses shape","classified"),
          "price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("comfortable enough but way too expensive for what it is",
     _gt({"comfort": ("just right","classified"), "price_value": ("too expensive","classified"),
          "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("perfect firmness and the price is actually very reasonable",
     _gt({"comfort": ("just right","classified"), "price_value": ("good value","classified"),
          "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("great loft, very comfortable, worth every cent",
     _gt({"comfort": ("just right","classified"), "shape": ("just right thickness","classified"),
          "price_value": ("worth it","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("too soft and too expensive, double disappointment",
     _gt({"comfort": ("too soft","classified"), "price_value": ("too expensive","classified"),
          "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("the pillow is too flat and went even flatter after one month",
     _gt({"shape": ("too thin","classified"), "durability": ("degrades quickly","classified"),
          "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("incredibly durable and comfortable, best pillow I've ever owned",
     _gt({"comfort": ("just right","classified"), "durability": ("lasts well","classified"),
          "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("comfortable, size is not bad, price is too much though",
     _gt({"comfort": ("just right","classified"), "shape": ("just right thickness","classified"),
          "price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Intent: needs reply ───────────────────────────────────────────────────
    ("does this come in a king size version?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("is this suitable for people who sleep on their side?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("can I wash this in a hot water cycle?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("do you offer bulk discounts for buying multiple pillows?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("what's the return policy if I don't find it comfortable?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("is this pillow hypoallergenic? my daughter has allergies",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("how long does delivery take to Bangkok?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("does it come with a pillowcase or just the pillow itself?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    # ── Intent: comparison ────────────────────────────────────────────────────
    ("much better than my old memory foam one, this is an upgrade",
     _gt({"intent": ("comparison","classified"), "tone": ("happy","classified")})),

    ("similar quality to the Tempur brand but at a much better price",
     _gt({"intent": ("comparison","classified"), "price_value": ("good value","classified"), "tone": ("happy","classified")})),

    ("not as good as the one I had before, that one lasted three years",
     _gt({"intent": ("comparison","classified"), "durability": ("degrades quickly","classified"), "tone": ("disappointed","classified")})),

    ("tried five different pillows before this, by far the best one",
     _gt({"intent": ("comparison","classified"), "tone": ("happy","classified")})),

    # ── Intent: spam ─────────────────────────────────────────────────────────
    ("check out my page for the best deals on home products!",
     _gt({"intent": ("spam","classified"), "tone": ("neutral","classified")})),

    ("visit our website for similar products at lower prices guaranteed",
     _gt({"intent": ("spam","classified"), "tone": ("neutral","classified")})),

    ("great post! also our shop is having a 50% sale this weekend",
     _gt({"intent": ("spam","classified"), "tone": ("neutral","classified")})),

    # ── Edge cases: sarcasm (often lands as Unclear) ──────────────────────────
    ("oh sure, because who needs to sleep comfortably anyway",
     _gt({"comfort": ("too firm","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("great pillow, really love waking up with a stiff neck every morning",
     _gt({"comfort": ("too firm","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("fantastic quality, only fell apart after two whole months!",
     _gt({"durability": ("degrades quickly","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    # ── Edge cases: very short ────────────────────────────────────────────────
    ("love it!",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("terrible.",
     _gt({"intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("too flat.",
     _gt({"shape": ("too thin","classified"), "intent": ("negative review","classified"), "tone": ("neutral","classified")})),

    ("perfect!",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("not bad.",
     _gt({"intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    ("overpriced.",
     _gt({"price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("neutral","classified")})),

    # ── Edge cases: indirect/niche language ───────────────────────────────────
    ("my cat has claimed this as her own, she sleeps on it every day",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("my partner keeps stealing it from my side of the bed",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("the pillow survived a house move and still works perfectly",
     _gt({"durability": ("lasts well","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("bought one for my mum and she loves it, ordering another",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("gave it as a gift and my friend said it was the best gift ever",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Tone focused ──────────────────────────────────────────────────────────
    ("I am absolutely furious, this is not what I paid for at all",
     _gt({"intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("a little let down if I'm honest, expected something better",
     _gt({"intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("curious whether the firm version would suit me better actually",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("delivered on time, dimensions are as listed, no issues",
     _gt({"intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("this is genuinely the best pillow purchase I have ever made",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Comfort + tone combinations ───────────────────────────────────────────
    ("the firmness is perfect, I'm genuinely delighted with this",
     _gt({"comfort": ("just right","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("too soft for my liking but I can see it working for some people",
     _gt({"comfort": ("too soft","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("so disappointed, paid good money and it's already going soft",
     _gt({"comfort": ("changes over time","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── Shape + price combinations ────────────────────────────────────────────
    ("flat, thin, and still charges premium prices? unbelievable",
     _gt({"shape": ("too thin","classified"), "price_value": ("too expensive","classified"),
          "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("great thickness for the price, really good deal",
     _gt({"shape": ("just right thickness","classified"), "price_value": ("good value","classified"),
          "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Neutral informational ─────────────────────────────────────────────────
    ("arrived in good condition, packaging was professional",
     _gt({"intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("standard pillow, does its job without anything special to note",
     _gt({"intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("used it for two weeks, nothing exceptional to report either way",
     _gt({"intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("the dimensions match the product listing exactly",
     _gt({"shape": ("just right thickness","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    # ── Mixed sentiment within one comment ────────────────────────────────────
    ("the comfort is excellent but the durability is terrible",
     _gt({"comfort": ("just right","classified"), "durability": ("degrades quickly","classified"),
          "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("pricey but comfortable, I'll let you know about durability later",
     _gt({"comfort": ("just right","classified"), "price_value": ("too expensive","classified"),
          "durability": ("too early to tell","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("soft and comfortable but already losing shape after three weeks",
     _gt({"comfort": ("just right","classified"), "shape": ("loses shape","classified"),
          "durability": ("degrades quickly","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    # ── More intent: needs reply ──────────────────────────────────────────────
    ("anyone know if this comes in a firmer version?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("is this suitable for children aged 8 to 12?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    ("can you recommend which size for a standard single bed?",
     _gt({"intent": ("needs reply","classified"), "tone": ("curious","classified")})),

    # ── More comparisons ──────────────────────────────────────────────────────
    ("not as thick as the photos suggested, different from what I expected",
     _gt({"shape": ("too thin","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("holds its shape better than any pillow I have owned before",
     _gt({"shape": ("just right thickness","classified"), "durability": ("lasts well","classified"),
          "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    # ── Final batch: varied ───────────────────────────────────────────────────
    ("I genuinely cannot sleep without this pillow now",
     _gt({"comfort": ("just right","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("the filling keeps shifting to one end, really annoying",
     _gt({"shape": ("loses shape","classified"), "intent": ("negative review","classified"), "tone": ("disappointed","classified")})),

    ("after three washes it's still holding up perfectly",
     _gt({"durability": ("lasts well","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("bought two and both are great, consistent quality",
     _gt({"durability": ("lasts well","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("not worth the sale price even, very poor product",
     _gt({"price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("perfect for back sleepers, less ideal for side sleepers",
     _gt({"shape": ("just right thickness","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("been using it for six months and no issues at all so far",
     _gt({"durability": ("lasts well","classified"), "intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    ("this is my third order, says everything you need to know",
     _gt({"intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("the price went up since I last bought it, still worth it though",
     _gt({"price_value": ("worth it","classified"), "intent": ("positive review","classified"), "tone": ("neutral","classified")})),

    ("really comfortable but I do wish it came in a firmer option",
     _gt({"comfort": ("just right","classified"), "intent": ("needs reply","classified"), "tone": ("happy","classified")})),

    ("soft, flat, overpriced — the trifecta of disappointment",
     _gt({"comfort": ("too soft","classified"), "shape": ("too thin","classified"),
          "price_value": ("too expensive","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("genuinely life-changing for my sleep quality, no exaggeration",
     _gt({"comfort": ("just right","classified"), "intent": ("positive review","classified"), "tone": ("happy","classified")})),

    ("arrived damaged, the cover was torn on arrival",
     _gt({"durability": ("degrades quickly","classified"), "intent": ("negative review","classified"), "tone": ("angry","classified")})),

    ("no strong feelings either way, it's just a pillow",
     _gt({"intent": ("monitor","classified"), "tone": ("neutral","classified")})),

    ("hard to say after only one use, seems okay",
     _gt({"durability": ("too early to tell","classified"), "intent": ("monitor","classified"), "tone": ("neutral","classified")})),
]


# ── Generator ─────────────────────────────────────────────────────────────────

def generate_synthetic_dataset(output_dir: str = "data") -> tuple[Path, Path]:
    """Generate synthetic dataset and ground truth files.

    Returns:
        (comments_path, ground_truth_path)
    """
    Path(output_dir).mkdir(exist_ok=True)

    comments = []
    ground_truth = []

    for i, (message, gt) in enumerate(COMMENTS_RAW):
        comment_id = f"c{i+1:03d}"
        comments.append({
            "id": comment_id,
            "idx": i,
            "message": message,
            "created_time": "2025-01-15T00:00:00",
            "like_count": 0,
            "comment_count": 0,
        })
        ground_truth.append({
            "comment_id": comment_id,
            "message": message,
            "ground_truth": gt,
        })

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_comments": len(comments),
        "dimensions": ALL_DIMS,
        "description": "Synthetic pillow shop comment dataset for Local Tailor demo",
    }

    comments_data = {"metadata": metadata, "comments": comments}
    gt_data = {"metadata": metadata, "ground_truth": ground_truth}

    comments_path = Path(output_dir) / "comments_clean_demo.json"
    gt_path = Path(output_dir) / "ground_truth_demo.json"

    with open(comments_path, "w", encoding="utf-8") as f:
        json.dump(comments_data, f, ensure_ascii=False, indent=2)

    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, ensure_ascii=False, indent=2)

    # Print distribution summary
    print(f"\n  ── Synthetic Dataset Summary ────────────────────────")
    print(f"  Total comments: {len(comments)}")
    for dim in ALL_DIMS:
        value_counts: Dict = {}
        for item in ground_truth:
            v = item["ground_truth"][dim]["value"]
            value_counts[v] = value_counts.get(v, 0) + 1
        non_na = {k: v for k, v in value_counts.items() if k != "N/A"}
        na_count = value_counts.get("N/A", 0)
        print(f"  {dim:15s}: {sum(non_na.values())} classified, {na_count} N/A")
    print(f"  ─────────────────────────────────────────────────────\n")

    return comments_path, gt_path


if __name__ == "__main__":
    generate_synthetic_dataset()
    print("Done.")
