"""Diagnose contour seam: overlay owner-mask boundaries on the LSDA/native output."""
import numpy as np
from PIL import Image, ImageFilter

RUN = "/science/wx/pry/MMDIT/experiments/2026_9_12_EXP_1_DEMO_P1/lsda_runs/runs/demo_p1_bluewhite_maiolica_s42_lsda_v1"
OUT = "/science/wx/pry/MMDIT/experiments/2026_9_12_EXP_1_DEMO_P1/figures"

ss = Image.open(RUN + "/native_ss.png").convert("RGB")
ls = Image.open(RUN + "/lsda.png").convert("RGB")
o1 = np.asarray(Image.open(RUN + "/segmentation/owner_1.png").convert("L")) > 127
o2 = np.asarray(Image.open(RUN + "/segmentation/owner_2.png").convert("L")) > 127


def boundary(mask):
    m = Image.fromarray((mask * 255).astype(np.uint8))
    er = np.asarray(m.filter(ImageFilter.MinFilter(5))) > 127
    return mask & ~er


b1 = boundary(o1)
b2 = boundary(o2)

x0, x1 = 500, 1024
w = x1 - x0
h = ls.size[1]


def crop(img):
    return img.crop((x0, 0, x1, h))


ss_c, ls_c = crop(ss), crop(ls)

arr = np.asarray(ls_c).copy()
arr[b1[:, x0:x1]] = [0, 90, 255]
arr[b2[:, x0:x1]] = [255, 0, 0]
ls_marked = Image.fromarray(arr)

canvas = Image.new("RGB", (w * 3, h), (255, 255, 255))
canvas.paste(ss_c, (0, 0))
canvas.paste(ls_c, (w, 0))
canvas.paste(ls_marked, (2 * w, 0))
canvas.save(OUT + "/diag_right_boundary.png")

lower = canvas.crop((0, 650, w * 3, h))
lower.save(OUT + "/diag_right_lower.png")
print("saved", OUT + "/diag_right_boundary.png")
