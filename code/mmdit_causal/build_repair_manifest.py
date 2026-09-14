"""Build a pair-mode scoring manifest for LSDA repair runs.

Scans <repair-root>/runs/*/lsda.png; run_id format lsda_{kind}_p{pair}_s{seed}.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pairs100 import a_only_prompt, b_only_prompt, ss_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair-root", type=Path)
    ap.add_argument("--jpg-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    records = []
    if args.jpg_dir is not None:
        candidates = [(p.stem, p) for p in sorted(args.jpg_dir.glob("lsda_*.jpg"))]
    else:
        candidates = [
            (d.name, d / "lsda.png")
            for d in sorted((args.repair_root / "runs").glob("lsda_*"))
            if (d / "lsda.png").exists()
        ]
    for run_id, image in candidates:
        m = re.match(r"lsda_(me|ka|uniform|meattrs)_p(\d+)_s(\d+)$", run_id)
        if not m:
            continue
        pair = int(m.group(2))
        records.append(
            {
                "id": run_id,
                "image": str(image).replace("\\", "/"),
                "mode": "pair",
                "scene": ss_prompt(pair),
                "left": a_only_prompt(pair),
                "right": b_only_prompt(pair),
                "kind": m.group(1),
                "pair": pair,
                "seed": int(m.group(3)),
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "out": str(args.out)}))


if __name__ == "__main__":
    main()
