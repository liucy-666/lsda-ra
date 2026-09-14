"""Assemble the 4-panel demo figure: Standalone A | Standalone B | Native SS | LSDA."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--standalone-a", required=True)
    ap.add_argument("--standalone-b", required=True)
    ap.add_argument("--ss", required=True)
    ap.add_argument("--lsda", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=640)
    args = ap.parse_args()

    panels = [
        ("Standalone A", args.standalone_a),
        ("Standalone B", args.standalone_b),
        ("Native SS", args.ss),
        ("LSDA", args.lsda),
    ]
    tile = args.tile
    bar = 56
    canvas = Image.new("RGB", (tile * len(panels), tile + bar), (250, 246, 238))
    draw = ImageDraw.Draw(canvas)
    font = load_font(30)
    for i, (label, path) in enumerate(panels):
        img = Image.open(path).convert("RGB").resize((tile, tile), Image.LANCZOS)
        canvas.paste(img, (i * tile, bar))
        draw.text((i * tile + 16, 14), label, fill=(31, 78, 121), font=font)
        draw.line([(i * tile, 0), (i * tile, tile + bar)], fill=(200, 200, 200), width=2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=95)
    print(str(out))


if __name__ == "__main__":
    main()
