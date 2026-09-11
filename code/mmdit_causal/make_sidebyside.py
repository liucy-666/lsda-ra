"""Side-by-side comparison of two images with labels."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", type=Path, required=True)
    ap.add_argument("--right", type=Path, required=True)
    ap.add_argument("--labels", nargs=2, default=["baseline", "v_fix"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    a = Image.open(args.left).convert("RGB")
    b = Image.open(args.right).convert("RGB")
    w, h = a.size
    pad = 16
    label_h = 40
    font = load_font(26)
    canvas = Image.new("RGB", (2 * w + 3 * pad, h + label_h + 2 * pad), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad + 8, 8), args.labels[0], fill="black", font=font)
    draw.text((2 * pad + w + 8, 8), args.labels[1], fill="black", font=font)
    canvas.paste(a, (pad, label_h))
    canvas.paste(b, (2 * pad + w, label_h))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"saved {args.out} {canvas.size}")


if __name__ == "__main__":
    main()
