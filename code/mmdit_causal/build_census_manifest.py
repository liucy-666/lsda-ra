"""Build the census scoring manifest (single A/B + pair SS) for score_images.py."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pairs100 import a_only_prompt, b_only_prompt, ss_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ext", default="png")
    ap.add_argument("--pairs", type=int, nargs="+", default=list(range(1, 101)))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1011, 1012, 1013])
    args = ap.parse_args()

    records = []
    for idx in args.pairs:
        a, b, ss = a_only_prompt(idx), b_only_prompt(idx), ss_prompt(idx)
        for seed in args.seeds:
            for kind, rec in (
                ("A", {"mode": "single", "desc": a}),
                ("B", {"mode": "single", "desc": b}),
                ("SS", {"mode": "pair", "scene": ss, "left": a, "right": b}),
            ):
                image = args.images_dir / f"pair{idx:03d}_seed{seed}_{kind}.{args.ext}"
                records.append(
                    {
                        "id": f"cen_p{idx:03d}_s{seed}_{kind}",
                        "image": str(image).replace("\\", "/"),
                        **rec,
                    }
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(__import__("json").dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    print(f"records={len(records)} out={args.out}")


if __name__ == "__main__":
    main()
