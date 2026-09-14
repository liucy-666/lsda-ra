"""Tier-1 texture metric: LBP histogram + GLCM features, region vs standalone reference.

Per sample and arm, drift = correct iff left region is closer to ref A and right to ref B.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

sys.path.insert(0, str(Path(__file__).parent))
from arms import load_classification, refs_for, resolve  # noqa: E402

GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
SIZE = 256


def center_crop(pil: Image.Image, frac: float = 0.7) -> Image.Image:
    w, h = pil.size
    bw, bh = int(w * frac), int(h * frac)
    x0, y0 = (w - bw) // 2, (h - bh) // 2
    return pil.crop((x0, y0, x0 + bw, y0 + bh))


def descriptors(pil: Image.Image):
    arr = np.array(center_crop(pil).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8)
    hists, feats = [], []
    for c in range(3):
        ch = np.ascontiguousarray(arr[:, :, c])
        lbp = local_binary_pattern(ch, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
        hists.append(hist)
        glcm = graycomatrix(
            ch, distances=[1, 2], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256, symmetric=True, normed=True,
        )
        feats.append(np.concatenate([graycoprops(glcm, p).ravel() for p in GLCM_PROPS]))
    feat = np.concatenate(feats)
    feat = feat / (np.linalg.norm(feat) + 1e-8)
    return np.concatenate(hists), feat


def dist(reg, ref):
    lbp_d = float(np.abs(reg[0] - ref[0]).sum())
    glcm_d = float(np.linalg.norm(reg[1] - ref[1]))
    return lbp_d + glcm_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", type=Path, required=True)
    ap.add_argument("--census-dir", type=Path, required=True)
    ap.add_argument("--lsda-dir", type=Path, required=True)
    ap.add_argument("--binding-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = load_classification(args.classification)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    out = []
    for rec in rows:
        arms = resolve(rec, args.census_dir, args.lsda_dir, args.binding_dir)
        a_ref, b_ref = refs_for(rec, args.census_dir)
        if not (a_ref.exists() and b_ref.exists()):
            continue
        d_a, d_b = descriptors(Image.open(a_ref)), descriptors(Image.open(b_ref))
        for arm, path in arms.items():
            if not path.exists():
                continue
            img = Image.open(path).convert("RGB")
            w, h = img.size
            left = descriptors(img.crop((0, 0, w // 2, h)))
            right = descriptors(img.crop((w // 2, 0, w, h)))
            dLA, dLB = dist(left, d_a), dist(left, d_b)
            dRA, dRB = dist(right, d_a), dist(right, d_b)
            correct = (dLA < dLB) and (dRB < dRA)
            out.append({
                "id": f"{arm}_p{rec['pair']:03d}_s{rec['seed']}", "arm": arm,
                "pair": int(rec["pair"]), "seed": int(rec["seed"]), "label": rec["label"],
                "correct": bool(correct), "dL_A": dLA, "dL_B": dLB, "dR_A": dRA, "dR_B": dRB,
            })
    with args.out.open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"texture rows={len(out)} in {time.time()-now:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
