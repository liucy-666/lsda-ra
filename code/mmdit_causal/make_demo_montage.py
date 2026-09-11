"""Build a demo montage: rows = seeds, columns = arms (baseline/w_fix/v_fix/both_fix)."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ARMS = ["baseline", "w_fix", "v_fix", "both_fix"]
ARM_LABEL = {
    "baseline": "baseline (drift)",
    "w_fix": "W-fix (routing)",
    "v_fix": "V-fix (content)",
    "both_fix": "Both-fix",
}


def load_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--pair", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    font = load_font(int(22 * args.scale))
    title_font = load_font(int(26 * args.scale))
    pad = int(12 * args.scale)
    label_h = int(34 * args.scale)

    tiles = {}
    w = h = None
    for seed in args.seeds:
        for arm in ARMS:
            p = args.dir / f"pair{args.pair}_seed{seed}_{arm}.png"
            if not p.exists():
                raise FileNotFoundError(p)
            img = Image.open(p).convert("RGB")
            if args.scale != 1.0:
                img = img.resize((int(img.width * args.scale), int(img.height * args.scale)), Image.LANCZOS)
            tiles[(seed, arm)] = img
            w, h = img.size

    n_rows, n_cols = len(args.seeds), len(ARMS)
    canvas = Image.new("RGB", (n_cols * w + (n_cols + 1) * pad, label_h + n_rows * (h + label_h) + (n_rows + 1) * pad), "white")
    draw = ImageDraw.Draw(canvas)

    for j, arm in enumerate(ARMS):
        x = pad + j * (w + pad)
        draw.text((x + 6, int(6 * args.scale)), ARM_LABEL[arm], fill="black", font=title_font)

    for i, seed in enumerate(args.seeds):
        y = label_h + pad + i * (h + label_h + pad)
        draw.text((pad, y), f"seed {seed}", fill="black", font=font)
        for j, arm in enumerate(ARMS):
            x = pad + j * (w + pad)
            canvas.paste(tiles[(seed, arm)], (x, y + label_h))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"saved {args.out} ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    main()
