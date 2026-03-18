"""
localtailor/config.py
=====================
DimensionConfig dataclass, YAML loader, and shop switching.

Shop data lives in shops/{SHOP}/:
  shops/{SHOP}/dimensions.yaml  — structure and descriptions
  shops/{SHOP}/examples.json    — training examples

Change SHOP to switch between shops (e.g. "pillow", "shoe").
All paths (data, models, config) are scoped by shop automatically.
"""

from __future__ import annotations
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
import yaml


# ── Shop registry ────────────────────────────────────────────────────────────

class Shop(str, Enum):
    """Built-in shops. Add new entries here after creating shops/{name}/."""
    PILLOW = "pillow"
    SHOE   = "shoe"

# ── Active shop ──────────────────────────────────────────────────────────────
# Change this to switch shops. All paths resolve from this.

SHOP = Shop.PILLOW

# ── Shop path resolver ───────────────────────────────────────────────────────

def shop_paths(shop: str | Shop | None = None) -> Dict[str, str]:
    """Return resolved paths for the given shop (defaults to SHOP).

    Returns dict with keys:
        shop          — shop name
        dimensions    — path to dimensions.yaml
        examples      — path to examples.json
        data_dir      — path to data output directory
        models_dir    — path to trained models directory
    """
    s = shop or SHOP
    return {
        "shop": s,
        "dimensions": f"shops/{s}/dimensions.yaml",
        "examples": f"shops/{s}/examples.json",
        "data_dir": f"data/{s}",
        "models_dir": f"models/{s}",
    }


@dataclass
class DimensionValue:
    label: str
    description: Optional[str] = None
    examples: List[str] = field(default_factory=list)

    def span_question(self, dimension_name: str) -> str:
        return f"What does this comment say about {dimension_name}?"


@dataclass
class DimensionConfig:
    name: str
    values: List[DimensionValue]
    enabled: bool = True

    def value_labels(self) -> List[str]:
        return [v.label for v in self.values]

    def all_examples(self) -> List[tuple[str, str]]:
        """Returns [(text, label), ...] for all values. Used by SetFit trainer."""
        return [(ex, v.label) for v in self.values for ex in v.examples]

    def min_examples_per_class(self) -> int:
        if not self.values:
            return 0
        return min(len(v.examples) for v in self.values)

    def __repr__(self):
        return (f"DimensionConfig(name='{self.name}', "
                f"values={self.value_labels()}, "
                f"min_examples={self.min_examples_per_class()})")


def load_dimensions(
    config_path: str | None = None,
    examples_path: str | None = None,
) -> List[DimensionConfig]:
    """Load dimensions from YAML and inject examples from examples.json.

    Args:
        config_path:   Path to dimensions.yaml. Defaults to shops/{SHOP}/dimensions.yaml.
        examples_path: Path to examples.json. Defaults to shops/{SHOP}/examples.json.

    Returns:
        List of DimensionConfig with examples populated.
    """
    paths = shop_paths()
    config_path = config_path or paths["dimensions"]
    examples_path = examples_path or paths["examples"]

    # Load YAML structure
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not raw or "dimensions" not in raw:
        raise ValueError("Config must have a top-level 'dimensions' key.")

    # Load examples (optional — if file doesn't exist, examples are empty)
    examples_data: Dict[str, Dict[str, List[str]]] = {}
    ex_path = Path(examples_path)
    if ex_path.exists():
        with open(ex_path, "r", encoding="utf-8") as f:
            raw_examples = json.load(f)
        # Strip metadata keys starting with _
        examples_data = {k: v for k, v in raw_examples.items() if not k.startswith("_")}
    else:
        print(f"  [config] No examples file found at {ex_path}. "
              f"SetFit will train with empty examples — create {ex_path} to add training data.")

    dimensions: List[DimensionConfig] = []
    seen_names: set = set()

    for i, dim_raw in enumerate(raw["dimensions"]):
        if "name" not in dim_raw:
            raise ValueError(f"Dimension at index {i} missing 'name'.")
        if "values" not in dim_raw or len(dim_raw["values"]) < 2:
            raise ValueError(f"Dimension '{dim_raw.get('name', i)}' needs at least 2 values.")

        name = dim_raw["name"].strip().lower().replace(" ", "_")
        if name in seen_names:
            raise ValueError(f"Duplicate dimension name: '{name}'.")
        seen_names.add(name)

        if not dim_raw.get("enabled", True):
            print(f"  [config] Skipping disabled dimension: '{name}'")
            continue

        dim_examples = examples_data.get(name, {})

        values: List[DimensionValue] = []
        for v in dim_raw["values"]:
            if isinstance(v, str):
                label = v.strip()
                values.append(DimensionValue(
                    label=label,
                    examples=dim_examples.get(label, []),
                ))
            elif isinstance(v, dict):
                label = v.get("label", "").strip()
                if not label:
                    raise ValueError(f"Dimension '{name}': value missing 'label'.")
                values.append(DimensionValue(
                    label=label,
                    description=v.get("description"),
                    examples=dim_examples.get(label, []),
                ))

        dimensions.append(DimensionConfig(name=name, values=values))

    if not dimensions:
        raise ValueError("No enabled dimensions found.")

    for d in dimensions:
        min_ex = d.min_examples_per_class()
        if min_ex < 8:
            print(f"  [config] WARNING: '{d.name}' has only {min_ex} examples/class "
                  f"(8 recommended). Add more in {examples_path}.")

    return dimensions


if __name__ == "__main__":
    if len(sys.argv) > 1:
        SHOP = Shop(sys.argv[1])
    print(f"\nAvailable shops: {', '.join(s.value for s in Shop)}")
    dims = load_dimensions()
    print(f"Active shop: {SHOP}")
    print(f"Loaded {len(dims)} dimension(s):\n")
    for d in dims:
        print(f"  {d}")
        for v in d.values:
            print(f"    • {v.label:30s} ({len(v.examples)} examples)")
    print()
