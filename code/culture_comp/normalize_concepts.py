"""Normalize CUBE-1K and TU-Darmstadt gold_concepts into one concept pool.

Output: benchmark/concepts_all.jsonl with
  {"key","concept","culture","category","source","lang","meta"}
CUBE source keeps its `name`,`country`,`domain`; TU translated keeps `concept`,`country`,`semantic_field`.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CUBE_CAT = {"cuisine": "food", "art": "art", "landmarks": "landmark", "landscapes": "landscape"}
TU_CAT = {
    "food": "food", "fruit": "food", "vegetable": "food", "vegetables": "food",
    "beverages": "food", "utensil": "utensil", "visual arts": "art", "clothing": "clothing",
    "music": "music", "musics": "music", "houses": "house", "celebration": "celebration",
    "sport": "sport", "sports": "sport", "religion_setting": "religion",
    "religion_object": "religion", "religion_custom": "religion",
}
TU_CULTURE = {"china": "China", "germany": "Germany", "south korea": "South Korea", "spain": "Spain"}


def norm_culture(c: str) -> str:
    c = c.strip()
    return TU_CULTURE.get(c.lower(), c.title())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", type=Path, required=True)
    ap.add_argument("--tu", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    cube = json.loads(args.cube.read_text(encoding="utf-8"))
    for i, r in enumerate(cube):
        dom = r.get("domain", "")
        rows.append({
            "key": f"CUBE_{i:04d}", "concept": r["name"], "culture": norm_culture(r["country"]),
            "category": CUBE_CAT.get(dom, dom), "source": "CUBE", "lang": "en",
            "meta": {"domain": dom, "cube_id": r.get("id")},
        })
    with args.tu.open(encoding="utf-8", newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            field = r["semantic_field"].strip()
            rows.append({
                "key": f"TU_{i:04d}", "concept": r["concept"].strip(),
                "culture": norm_culture(r["country"]), "category": TU_CAT.get(field, field),
                "source": "TU", "lang": r.get("language", "").strip(),
                "meta": {"semantic_field": field},
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    print("total:", len(rows))
    print("by source:", dict(collections.Counter(r["source"] for r in rows)))
    print("by category:", dict(collections.Counter(r["category"] for r in rows)))
    print("max tokens/obj:", max(len(r["concept"].split()) for r in rows))
    print(str(args.out))


if __name__ == "__main__":
    main()
