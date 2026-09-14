"""Rule-based extraction of KA knowledge phrases from the 100-pair long descriptions.

For each entity, split the long description into clauses and keep the attribute-bearing
clauses (material / palette / technique / motif), producing a short phrase list that can be
appended to the entity's short prompt when regenerating a knowledge-absent region.
"""
import argparse
import json
import re
from pathlib import Path

SPLIT = re.compile(r",\s*|\s+with\s+|\s+made of\s+|\s+made from\s+|\s+showing\s+|\s+painted\s+|\s+decorated\s+|\s+worked\s+|\s+woven\s+", re.I)
LEAD = re.compile(r"^(and|showing|with|painted|in|on|of|decorated|worked|woven|made of|made from|the)\s+", re.I)

KEYS = (
    "glaze", "enamel", "paint", "decor", "motif", "pattern", "color", "colour",
    "blue", "green", "red", "yellow", "gold", "silver", "white", "black", "brown",
    "purple", "turquoise", "ochre", "manganese", "cobalt", "pink", "orange", "cream",
    "porcelain", "stoneware", "earthenware", "faience", "bronze", "lacquer", "silk",
    "wool", "cotton", "clay", "tin", "copper", "iron", "steel", "jade", "wood",
    "carv", "inlay", "weav", "woven", "dye", "relief", "gloss", "matte", "crackle",
    "gild", "lustre", "luster", "cast", "hammered", "incised", "polychrome", "overglaze",
    "underglaze", "floral", "geometric", "scroll", "arabesque", "medallion", "figure",
    "landscape", "dragon", "lotus", "bird", "animal", "palace", "scene", "border",
)


def clean_clause(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = LEAD.sub("", text).strip(" .,")
    return text


def extract(long_desc: str, max_phrases: int = 8) -> str:
    clauses = [clean_clause(c) for c in SPLIT.split(long_desc) if c.strip(" .")]
    clauses = [c for c in clauses if len(c) >= 3]
    if not clauses:
        return ""
    kept = [c for c in clauses[1:] if any(k in c.lower() for k in KEYS)]
    if not kept:
        kept = clauses[1:]
    seen, out = set(), []
    for c in kept:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return "; ".join(out[:max_phrases])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = json.loads(args.pairs_json.read_text(encoding="utf-8"))
    out = {}
    for i, rec in enumerate(data, start=1):
        a_long = rec["文化物体A的长文本描述"]
        b_long = rec["文化物体B的长文本描述"]
        out[f"{i:03d}"] = {
            "A": {"short": rec["文化物体A"], "long": a_long, "attrs": extract(a_long)},
            "B": {"short": rec["文化物体B"], "long": b_long, "attrs": extract(b_long)},
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"pairs": len(out), "out": str(args.out)}, ensure_ascii=False))
    for k in ("001", "003", "015"):
        if k in out:
            print(f"pair {k} A: {out[k]['A']['attrs']}")
            print(f"pair {k} B: {out[k]['B']['attrs']}")


if __name__ == "__main__":
    main()
