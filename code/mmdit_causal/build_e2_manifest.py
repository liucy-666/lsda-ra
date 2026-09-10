"""Build an E2 scoring manifest from generated arm images."""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pairs import a_only_prompt, b_only_prompt, ss_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=["baseline", "w_fix", "v_fix", "both_fix"])
    args = ap.parse_args()

    records = []
    for p in sorted(args.dir.glob("*.png")):
        if "_donorB" in p.name:
            continue
        m = re.match(r"pair(\d+)_seed(\d+)_(.+)\.png$", p.name)
        if not m:
            continue
        pair, seed, arm = int(m.group(1)), int(m.group(2)), m.group(3)
        if arm not in args.arms:
            continue
        records.append({
            "id": f"e2_p{pair}_s{seed}_{arm}",
            "image": str(p).replace("\\", "/"),
            "mode": "pair",
            "scene": ss_prompt(pair),
            "left": a_only_prompt(pair),
            "right": b_only_prompt(pair),
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "out": str(args.out)}))


if __name__ == "__main__":
    main()
