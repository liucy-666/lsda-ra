"""Quantify discriminative margins of the objective metrics (are they usable?)."""
import json
from pathlib import Path

import numpy as np

R = Path(r"D:\Python\MMDIT\experiment\2026_9_12_EXP_3_KA_ME\ratings")


def load(name):
    return [json.loads(l) for l in (R / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def stats(rows, pre):
    dL, dR, okL, okR, both = [], [], 0, 0, 0
    for x in rows:
        l_a, l_b = x[f"{pre}_L_A"], x[f"{pre}_L_B"]
        r_a, r_b = x[f"{pre}_R_A"], x[f"{pre}_R_B"]
        dL.append(abs(l_a - l_b))
        dR.append(abs(r_a - r_b))
        okL += l_a > l_b
        okR += r_b > r_a
        both += (l_a > l_b) and (r_b > r_a)
    n = len(rows)
    print(f"{pre:8s} n={n}  mean|dL|={np.mean(dL):.4f}  mean|dR|={np.mean(dR):.4f}  "
          f"L-correct={okL / n:.3f}  R-correct={okR / n:.3f}  both-correct={both / n:.3f}")


clip = load("objective_clip.jsonl")
dino = load("objective_dino.jsonl")
stats(clip, "clip_i")
stats(clip, "clip_t")
stats(dino, "dino")
