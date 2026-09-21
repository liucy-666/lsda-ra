"""SS framing-fix comparison: rows = pairs, cols = current / T1 / T2 / T3.
Each cell = [native_ss | lsda(v15mask)] side by side."""
from pathlib import Path

from PIL import Image, ImageDraw

CUR = Path("D:/Python/MMDIT/data/Aesth_Pilot/2026_9_15_EXP_7")
FIX = Path("D:/Python/MMDIT/data/Aesth_Pilot/2026_9_15_EXP_7_fixframe/runs")
OUT = Path("D:/Python/MMDIT/experiment/2026_9_15_EXP_7_FOOD_PILOT/figures")
PAIRS = ["002", "014", "016"]
SUBS = 260
HDR = 30


def cell(ss: Path, lsda: Path) -> Image.Image:
    im = Image.new("RGB", (SUBS * 2, SUBS), (255, 255, 255))
    for k, p in enumerate((ss, lsda)):
        im.paste(Image.open(p).convert("RGB").resize((SUBS, SUBS), Image.Resampling.LANCZOS), (k * SUBS, 0))
    return im


def main() -> None:
    cols = ["current", "T1", "T2", "T3"]
    cw = SUBS * 2
    canvas = Image.new("RGB", (cw * len(cols), HDR + SUBS * len(PAIRS)), (250, 246, 238))
    d = ImageDraw.Draw(canvas)
    for c, n in enumerate(cols):
        d.text((c * cw + 8, 8), n, fill=(31, 78, 121))
    for r, p in enumerate(PAIRS):
        y = HDR + r * SUBS
        canvas.paste(cell(CUR / "SS" / f"p{p}_s1011.png", CUR / "v15mask" / f"p{p}_s1011.png"), (0, y))
        for c, t in enumerate(["T1", "T2", "T3"], start=1):
            run = FIX / f"fix{t}_p{p}_s1011"
            canvas.paste(cell(run / "native_ss.png", run / "lsda.png"), (c * cw, y))
        d.text((4, y + 4), f"p{p}", fill=(220, 40, 40))
    out = OUT / "fig_fixframe.png"
    canvas.save(out, quality=95)
    print(out)


if __name__ == "__main__":
    main()
