"""Build the Phase 4 scoring manifest for the deep-dive 1024 images.

Scans the local deep-dive image dir and emits JSONL records:
  SS  -> mode "pair"  (scene = SS prompt, left = A, right = B)
  A/B -> mode "single" (desc = entity)
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pairs100 import PAIRS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    records = []
    for p in sorted(args.dir.glob("*.png")):
        m = re.match(r"(pair_\d+)_seed(\d+)_(SS|A|B)\.png$", p.name)
        if not m:
            continue
        pair, seed, kind = m.group(1), int(m.group(2)), m.group(3)
        idx = int(pair.split("_")[1])
        ent = PAIRS[idx]
        rec = {"id": f"dd_{pair}_s{seed}_{kind}", "image": str(p).replace("\\", "/")}
        if kind == "SS":
            rec.update(mode="pair", scene=ent["SS"], left=ent["A"], right=ent["B"])
        elif kind == "A":
            rec.update(mode="single", desc=ent["A"])
        else:
            rec.update(mode="single", desc=ent["B"])
        records.append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "out": str(args.out)}))


if __name__ == "__main__":
    main()
