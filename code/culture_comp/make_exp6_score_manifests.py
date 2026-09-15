"""Build dual-VLM scoring manifests for EXP_6 (LL, LSDA-Long).

Same judging protocol as EXP_3 census: pair mode, left=short_A, right=short_B, scene=SS prompt.
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--img-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seeds", default="1011,1012,1013")
    args = ap.parse_args()

    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))
    seeds = [int(s) for s in args.seeds.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for arm, prefix in (("LL", "ll"), ("LSDA_Long", "lsdalong")):
        rows = []
        for i, rec in enumerate(pairs, start=1):
            for s in seeds:
                rid = f"{prefix}_p{i:03d}_s{s}"
                img = args.img_root / arm / f"{rid}.jpg"
                rows.append({
                    "id": rid, "image": str(img), "mode": "pair",
                    "scene": rec["组合Prompt SS"],
                    "left": rec["文化物体A"], "right": rec["文化物体B"],
                })
        out = args.out_dir / f"{prefix}_score_manifest.jsonl"
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        print(arm, len(rows), out)


if __name__ == "__main__":
    main()
