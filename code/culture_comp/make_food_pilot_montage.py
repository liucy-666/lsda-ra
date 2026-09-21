"""Build the food pilot montage: rows = pairs, cols = A / B / SS / LSDA_Attr."""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path("D:/Python/MMDIT/data/Food_Pilot/2026_9_15_EXP_7")
OUT = Path("D:/Python/MMDIT/experiment/2026_9_15_EXP_7_FOOD_PILOT/figures")
COLS = ["A", "B", "SS", "LSDA_Attr"]
PAIRS = ["p001_s42", "p002_s42", "p003_s42"]
S = 512
HDR = 34


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    w = S * len(COLS)
    h = HDR + S * len(PAIRS)
    canvas = Image.new("RGB", (w, h), (250, 246, 238))
    draw = ImageDraw.Draw(canvas)
    for c, name in enumerate(COLS):
        draw.text((c * S + 10, 10), name, fill=(31, 78, 121))
    for r, pid in enumerate(PAIRS):
        for c, arm in enumerate(COLS):
            p = ROOT / arm / f"{pid}.png"
            im = Image.open(p).convert("RGB").resize((S, S), Image.Resampling.LANCZOS)
            canvas.paste(im, (c * S, HDR + r * S))
        draw.text((10, HDR + r * S + 6), pid, fill=(220, 40, 40))
    out = OUT / "fig_food_pilot.png"
    canvas.save(out, quality=95)
    print(out)


if __name__ == "__main__":
    main()
