"""Diagnose the straight seam/line in LSDA v1.4 rect outputs.

For each pair: find vertical columns where LSDA has a strong persistent edge that native SS
does not, report their positions (mod 8 -> latent grid), colors, and save a center-crop
SS-vs-LSDA figure with the detected seams marked.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BASE = Path("D:/Python/MMDIT/data/Food_Pilot/2026_9_15_EXP_7")
OUT = Path("D:/Python/MMDIT/experiment/2026_9_15_EXP_7_FOOD_PILOT/figures")
PAIRS = ["p001_s42", "p002_s42", "p003_s42"]
CROP = 300  # center crop half-width for the zoom figure


def grad_x(a: np.ndarray) -> np.ndarray:
    d = np.abs(np.diff(a, axis=1)).sum(axis=2)  # (H, W-1)
    return d.mean(axis=0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    S = 400
    rows = []
    for pid in PAIRS:
        lsda = np.asarray(Image.open(BASE / "LSDA_Attr" / f"{pid}.png").convert("RGB")).astype(np.float32)
        ss = np.asarray(Image.open(BASE / "SS" / f"{pid}.png").convert("RGB")).astype(np.float32)
        H, W, _ = lsda.shape
        gl, gs = grad_x(lsda), grad_x(ss)
        excess = gl - gs
        top = np.argsort(excess)[::-1][:200]
        top = sorted(int(x) for x in top if int(W * 0.20) < x < int(W * 0.80))
        print(f"=== {pid} ===  W={W}")
        shown = 0
        for x in top:
            if shown >= 6:
                break
            rgb = tuple(int(v) for v in lsda[:, x + 1].mean(axis=0))
            print(f"   seam x={x} (mod8={x % 8})  excess={excess[x]:.1f}  RGB={rgb}")
            shown += 1
        # figure: SS | LSDA center crop, seam columns marked
        x0 = W // 2 - CROP
        x1 = W // 2 + CROP
        canvas = Image.new("RGB", (S * 2, S + 22), (250, 246, 238))
        dr = ImageDraw.Draw(canvas)
        for k, (lab, arr) in enumerate((("native SS", ss), ("LSDA", lsda))):
            crop = Image.fromarray(arr[:, x0:x1].astype(np.uint8)).resize((S, S), Image.Resampling.LANCZOS)
            canvas.paste(crop, (k * S, 22))
            dr.text((k * S + 8, 4), lab, fill=(31, 78, 121))
        canvas.save(OUT / f"seam_{pid}.png")
        rows.append(pid)
    print("saved seam_*.png")


if __name__ == "__main__":
    main()
