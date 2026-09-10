"""Smoke test for the scoring pipeline on sample_0001 (native_SS + lsda_clean)."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import score_binding as sb

SEG = sb.EXP / "segmentation"


def run(sid: str, cond: str, ent_a: str, ent_b: str) -> None:
    root = "SS" if cond == "native_SS" else "LSDA"
    cand = Image.open(rf"D:\Python\MMDIT\data\{root}\2026_8_25_EXP_1\{sid}.jpg").convert("RGB")
    a = Image.open(rf"D:\Python\MMDIT\data\Standalone_A\2026_8_25_EXP_1\{sid}.jpg").convert("RGB")
    b = Image.open(rf"D:\Python\MMDIT\data\Standalone_B\2026_8_25_EXP_1\{sid}.jpg").convert("RGB")
    m = np.load(SEG / f"masks_{cond}" / f"{sid}_mask.npz")
    L, R = m["left"], m["right"]
    am = np.load(SEG / "masks_standalone_A" / f"{sid}_mask.npz")["fg"]
    bm = np.load(SEG / "masks_standalone_B" / f"{sid}_mask.npz")["fg"]
    lt = sb.siglip_patch_tokens(sig, sp, cand, L)
    rt = sb.siglip_patch_tokens(sig, sp, cand, R)
    refA = {
        "siglip": sb.siglip_patch_tokens(sig, sp, a, am),
        "clip": sb.embed_clip_image(clip_m, clip_p, a),
        "dino": sb.embed_dino(dino_m, dino_p, a),
    }
    refB = {
        "siglip": sb.siglip_patch_tokens(sig, sp, b, bm),
        "clip": sb.embed_clip_image(clip_m, clip_p, b),
        "dino": sb.embed_dino(dino_m, dino_p, b),
    }
    lc = sb.crop_by_mask(cand, L)
    rc = sb.crop_by_mask(cand, R)
    cl = sb.embed_clip_image(clip_m, clip_p, lc)
    cr = sb.embed_clip_image(clip_m, clip_p, rc)
    dl = sb.embed_dino(dino_m, dino_p, lc)
    dr = sb.embed_dino(dino_m, dino_p, rc)
    tA = sb.embed_clip_text(clip_m, clip_p, ent_a)
    tB = sb.embed_clip_text(clip_m, clip_p, ent_b)
    print(f"--- {cond} {sid} ---")
    print(
        f"siglip_cp L_A={sb.masked_maxcos(refA['siglip'], lt):.3f} L_B={sb.masked_maxcos(refB['siglip'], lt):.3f}"
        f" | R_A={sb.masked_maxcos(refA['siglip'], rt):.3f} R_B={sb.masked_maxcos(refB['siglip'], rt):.3f}"
    )
    print(
        f"clip_i  L_A={sb.cos(cl, refA['clip']):.3f} L_B={sb.cos(cl, refB['clip']):.3f}"
        f" | R_A={sb.cos(cr, refA['clip']):.3f} R_B={sb.cos(cr, refB['clip']):.3f}"
    )
    print(
        f"clip_t  L_A={sb.cos(cl, tA):.3f} L_B={sb.cos(cl, tB):.3f}"
        f" | R_A={sb.cos(cr, tA):.3f} R_B={sb.cos(cr, tB):.3f}"
    )
    print(
        f"dino    L_A={sb.cos(dl, refA['dino']):.3f} L_B={sb.cos(dl, refB['dino']):.3f}"
        f" | R_A={sb.cos(dr, refA['dino']):.3f} R_B={sb.cos(dr, refB['dino']):.3f}"
    )


if __name__ == "__main__":
    try:
        print("loading models...", flush=True)
        sig, sp = sb.load_siglip2()
        clip_m, clip_p = sb.load_clip()
        dino_m, dino_p = sb.load_dino()
        run("sample_0001", "native_SS",
            "a Chinese blue-and-white porcelain vase",
            "a Vietnamese Bát Tràng blue-and-white ceramic vase")
        run("sample_0001", "lsda_clean",
            "a Chinese blue-and-white porcelain vase",
            "a Vietnamese Bát Tràng blue-and-white ceramic vase")
        print("SMOKE OK", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
