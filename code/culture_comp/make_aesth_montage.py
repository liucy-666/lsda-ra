"""Aesthetics comparison montage: rows = pairs, cols = SS / v14rect / v15mask / v15alpha."""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path("D:/Python/MMDIT/data/Aesth_Pilot/2026_9_15_EXP_7")
OUT = Path("D:/Python/MMDIT/experiment/2026_9_15_EXP_7_FOOD_PILOT/figures")
COLS = ["SS", "v14rect", "v15mask", "v15alpha"]
GROUPS = [["001", "002", "004", "008", "014"], ["016", "022", "041", "061", "081"]]
S = 420
HDR = 34


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for gi, pairs in enumerate(GROUPS, 1):
        w = S * len(COLS)
        h = HDR + S * len(pairs)
        canvas = Image.new("RGB", (w, h), (250, 246, 238))
        draw = ImageDraw.Draw(canvas)
        for c, name in enumerate(COLS):
            draw.text((c * S + 10, 10), name, fill=(31, 78, 121))
        for r, p in enumerate(pairs):
            for c, arm in enumerate(COLS):
                im = Image.open(ROOT / arm / f"p{p}_s1011.png").convert("RGB").resize((S, S), Image.Resampling.LANCZOS)
                canvas.paste(im, (c * S, HDR + r * S))
            draw.rectangle([0, HDR + r * S, w, HDR + r * S + 2], fill=(220, 40, 40))
            draw.text((6, HDR + r * S + 6), f"p{p}", fill=(31, 78, 121))
        out = OUT / f"fig_aesth_{gi}.png"
        canvas.save(out, quality=95)
        print(out)


if __name__ == "__main__":
    main()
