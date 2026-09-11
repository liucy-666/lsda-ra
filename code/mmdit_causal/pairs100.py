"""100 cultural pairs (cultural_pairs_100.json) with prompt builders (1-based index)."""
from __future__ import annotations

import json
from pathlib import Path

JSON_PATH = Path(__file__).parent / "cultural_pairs_100.json"


def _load() -> dict:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    out = {}
    for i, p in enumerate(data):
        out[i + 1] = {
            "A": p["文化物体A"],
            "B": p["文化物体B"],
            "A_long": p["文化物体A的长文本描述"],
            "B_long": p["文化物体B的长文本描述"],
            "SS": p["组合Prompt SS"],
        }
    return out


PAIRS = _load()


def ss_prompt(idx: int) -> str:
    return PAIRS[idx]["SS"]


def a_only_prompt(idx: int) -> str:
    return PAIRS[idx]["A"]


def b_only_prompt(idx: int) -> str:
    return PAIRS[idx]["B"]


def b_donor_prompt(idx: int) -> str:
    """B-only branch with the object at the SAME position (right) as in the mixed scene."""
    return f"Neutral studio background: {PAIRS[idx]['B']} on the right; fully visible."
