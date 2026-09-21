"""Demo montage: SS(no knowledge) vs LSDA(knowledge injection), with standalone A/B reference.

Fig 1 (fig_demo.png): rows = seeds, cols = standalone A / standalone B / SS / LSDA.
Fig 2 (fig_demo_best.png): the best-framed seed (lowest SS border contact), enlarged 4-panel.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("D:/Python/MMDIT/data/Demo_Doces_Litchi/2026_9_15_EXP_8")
OUT = Path("D:/Python/MMDIT/experiment/2026_9_15_EXP_8_DEMO_DOCES_LITCHI/figures")
COLS = ["standalone A (known.)", "standalone B (known.)", "SS (nouns only)", "LSDA (v15mask)"]
SEEDS = ["s42", "s43", "s44"]


def contact(p: Path) -> float:
    im = np.asarray(Image.open(p).convert("RGB").resize((128, 128))).astype(np.float32)
    border = np.concatenate([im[:5].reshape(-1, 3), im[-5:].reshape(-1, 3),
                             im[:, :5].reshape(-1, 3), im[:, -5:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    fg = np.linalg.norm(im - bg[None, None], axis=2) > 60
    return float(fg[:, :3].mean() + fg[:, -3:].mean())


def panel(paths, s, hdr=34):
    w = s * len(COLS)
    canvas = Image.new("RGB", (w, hdr + s), (250, 246, 238))
    d = ImageDraw.Draw(canvas)
    for c, (name, p) in enumerate(zip(COLS, paths)):
        im = Image.open(p).convert("RGB").resize((s, s), Image.Resampling.LANCZOS)
        canvas.paste(im, (c * s, hdr))
        d.text((c * s + 8, 8), name, fill=(31, 78, 121))
    return canvas


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sa, sb = ROOT / "SA" / "A.png", ROOT / "SB" / "B.png"

    # Fig 1
    s = 420
    hdr = 34
    canvas = Image.new("RGB", (s * len(COLS), hdr + s * len(SEEDS)), (250, 246, 238))
    d = ImageDraw.Draw(canvas)
    for c, name in enumerate(COLS):
        d.text((c * s + 8, 8), name, fill=(31, 78, 121))
    for r, seed in enumerate(SEEDS):
        paths = [sa, sb, ROOT / seed / "SS.png", ROOT / seed / "LSDA.png"]
        for c, p in enumerate(paths):
            im = Image.open(p).convert("RGB").resize((s, s), Image.Resampling.LANCZOS)
            canvas.paste(im, (c * s, hdr + r * s))
        d.text((6, hdr + r * s + 6), seed, fill=(220, 40, 40))
    (OUT / "fig_demo.png").parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT / "fig_demo.png", quality=95)
    print("fig_demo.png")

    # Fig 2 (best framing)
    scores = {seed: contact(ROOT / seed / "SS.png") for seed in SEEDS}
    best = min(scores, key=scores.get)
    print("SS contact:", {k: round(v, 3) for k, v in scores.items()}, "-> best", best)
    panel([sa, sb, ROOT / best / "SS.png", ROOT / best / "LSDA.png"], s=700).save(OUT / "fig_demo_best.png", quality=95)
    print("fig_demo_best.png", best)


if __name__ == "__main__":
    main()
