"""Selected-small comparison: rows = pairs, cols = old SS(seed1011) / new SS(selected) / new LSDA(v15mask)."""
from pathlib import Path

from PIL import Image, ImageDraw

OLD = Path("D:/Python/MMDIT/data/Aesth_Pilot/2026_9_15_EXP_7")
SEL = Path("D:/Python/MMDIT/data/Aesth_Pilot/2026_9_15_EXP_7_selsmall")
OUT = Path("D:/Python/MMDIT/experiment/2026_9_15_EXP_7_FOOD_PILOT/figures")
GROUPS = [["001", "002", "004", "008", "014"], ["016", "022", "041", "061", "081"]]
COLS = ["old SS", "new SS (selected)", "new LSDA (v15mask)"]
S = 420
HDR = 34


def find(d: Path, idx: str) -> Path:
    hits = sorted(d.glob(f"*_p{idx}_s*.png"))
    if not hits:
        raise FileNotFoundError(d / f"*p{idx}*")
    return hits[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for gi, pairs in enumerate(GROUPS, 1):
        w = S * len(COLS)
        h = HDR + S * len(pairs)
        canvas = Image.new("RGB", (w, h), (250, 246, 238))
        d = ImageDraw.Draw(canvas)
        for c, n in enumerate(COLS):
            d.text((c * S + 10, 10), n, fill=(31, 78, 121))
        for r, idx in enumerate(pairs):
            paths = [
                OLD / "SS" / f"p{idx}_s1011.png",
                find(SEL / "SS", idx),
                find(SEL / "LSDA", idx),
            ]
            for c, p in enumerate(paths):
                im = Image.open(p).convert("RGB").resize((S, S), Image.Resampling.LANCZOS)
                canvas.paste(im, (c * S, HDR + r * S))
            d.text((6, HDR + r * S + 6), f"p{idx}", fill=(220, 40, 40))
        out = OUT / f"fig_small_{gi}.png"
        canvas.save(out, quality=95)
        print(out)


if __name__ == "__main__":
    main()
