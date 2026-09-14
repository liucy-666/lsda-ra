"""Build a pair-mode scoring manifest for the attention-binding baseline."""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pairs100 import a_only_prompt, b_only_prompt, ss_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--ext", default="png")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    records = []
    for p in sorted(args.dir.glob(f"binding*.{args.ext}")):
        m = re.match(r"binding2?_p(\d+)_s(\d+)$", p.stem)
        if not m:
            continue
        pair, seed = int(m.group(1)), int(m.group(2))
        records.append({
            "id": f"bind_p{pair:03d}_s{seed}",
            "image": str(p).replace("\\", "/"),
            "mode": "pair",
            "scene": ss_prompt(pair),
            "left": a_only_prompt(pair),
            "right": b_only_prompt(pair),
            "pair": pair,
            "seed": seed,
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records)}))


if __name__ == "__main__":
    main()
