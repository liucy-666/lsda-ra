"""Diagnose latent-resolution (16x) mask quantization vs image-space owner masks.

Also reports |lsda - native| inside/outside each owner mask to localize artifacts.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

RUN = "/science/wx/pry/MMDIT/experiments/2026_9_12_EXP_1_DEMO_P1/lsda_runs/runs/demo_p1_bluewhite_maiolica_s42_lsda_v1"
OUT = "/science/wx/pry/MMDIT/experiments/2026_9_12_EXP_1_DEMO_P1/figures"

ss = np.asarray(Image.open(RUN + "/native_ss.png").convert("RGB"), dtype=np.float32)
ls = np.asarray(Image.open(RUN + "/lsda.png").convert("RGB"), dtype=np.float32)
o1 = np.asarray(Image.open(RUN + "/segmentation/owner_1.png").convert("L")) > 127
o2 = np.asarray(Image.open(RUN + "/segmentation/owner_2.png").convert("L")) > 127


def to_latent(m):
    t = torch.from_numpy(m.astype(np.float32))[None, None]
    return F.interpolate(t, size=(64, 64), mode="nearest")[0, 0].numpy() > 0.5


def up(m):
    return np.asarray(
        Image.fromarray((m * 255).astype(np.uint8)).resize((1024, 1024), Image.NEAREST)
    ) > 127


l1, l2 = to_latent(o1), to_latent(o2)
u1, u2 = up(l1), up(l2)

print("A area  image/latent/roundtrip:", round(float(o1.mean()), 4), round(float(l1.mean()), 4), round(float(u1.mean()), 4))
print("B area  image/latent/roundtrip:", round(float(o2.mean()), 4), round(float(l2.mean()), 4), round(float(u2.mean()), 4))
print("A mismatch (roundtrip vs image):", round(float((u1 ^ o1).mean()), 4))
print("B mismatch (roundtrip vs image):", round(float((u2 ^ o2).mean()), 4))

bg = ~(u1 | u2)
diff = np.abs(ls - ss).mean(2)
print("mean |lsda-native|  A:", round(float(diff[u1].mean()), 3),
      " B:", round(float(diff[u2].mean()), 3),
      " bg:", round(float(diff[bg].mean()), 5))

vis = np.zeros((1024, 1024, 3), np.uint8) + 255
vis[o2] = [255, 190, 190]
vis[u2 & ~o2] = [0, 0, 255]     # latent added (not in image mask)
vis[o2 & ~u2] = [180, 0, 0]     # image-mask pixels lost at latent res
Image.fromarray(vis).save(OUT + "/diag_latent_mask.png")

hm = np.clip(diff * 3, 0, 255).astype(np.uint8)
Image.fromarray(hm).save(OUT + "/diag_diff.png")
print("saved diag_latent_mask.png, diag_diff.png")
