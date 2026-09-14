"""SigLIP2 MaSC-style binding metric for the KA/ME census (single process, memory-light).

For each composition image: compute SigLIP2 patch tokens at 224x224, select left/right half
patch subsets, and compare against standalone A/B references via masked max-cosine (MaSC CP).
Attribution: left->A and right->B.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image

ROOT = Path(r"D:\Python\MMDIT")
MODELS = ROOT / "models"
DEV = "cpu"
torch.set_num_threads(max(1, (os.cpu_count() or 4)))


def load():
    from transformers import AutoProcessor, Siglip2VisionModel
    name = "google__siglip2-so400m-patch16-naflex"
    model = Siglip2VisionModel.from_pretrained(str(MODELS / name), torch_dtype=torch.float32).to(DEV).eval()
    proc = AutoProcessor.from_pretrained(str(MODELS / name))
    return model, proc


@torch.no_grad()
def patch_tokens(model, proc, pil, mask=None):
    s = 224
    img = pil.convert("RGB").resize((s, s), Image.BILINEAR)
    inputs = proc(images=img, return_tensors="pt", size={"height": s, "width": s}).to(DEV)
    out = model(
        pixel_values=inputs["pixel_values"],
        pixel_attention_mask=inputs["pixel_attention_mask"],
        spatial_shapes=inputs["spatial_shapes"],
    )
    tokens = out.last_hidden_state[0]
    tokens = tokens / tokens.norm(dim=-1, keepdim=True)
    if mask is not None:
        gh, gw = inputs["spatial_shapes"][0].tolist()
        grid = int(round((gh * gw) ** 0.5))
        m = Image.fromarray((mask * 255).astype(np.uint8)).resize((grid, grid), Image.NEAREST)
        m = np.asarray(m) > 127
        sel = m.reshape(-1)
        if sel.sum() == 0:
            return tokens
        tokens = tokens[sel]
    return tokens


def masked_maxcos(ref_tokens, cand_tokens):
    if ref_tokens.shape[0] == 0 or cand_tokens.shape[0] == 0:
        return float("nan")
    sims = ref_tokens @ cand_tokens.T
    return float(sims.max(dim=1).values.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", type=Path, required=True)
    ap.add_argument("--census-dir", type=Path, required=True)
    ap.add_argument("--repair-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cls = json.loads(args.classification.read_text(encoding="utf-8"))
    model, proc = load()
    print(f"siglip device={DEV}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])

    ref_cache = {}

    def refs(pair, seed):
        key = (pair, seed)
        if key in ref_cache:
            return ref_cache[key]
        a = Image.open(args.census_dir / f"pair{pair:03d}_seed{seed}_A.jpg").convert("RGB")
        b = Image.open(args.census_dir / f"pair{pair:03d}_seed{seed}_B.jpg").convert("RGB")
        ref_cache[key] = {"a": patch_tokens(model, proc, a), "b": patch_tokens(model, proc, b)}
        return ref_cache[key]

    def half_masks(w, h):
        left = np.zeros((h, w), np.uint8)
        left[:, : w // 2] = 1
        right = np.zeros((h, w), np.uint8)
        right[:, w // 2 :] = 1
        return left, right

    t0 = time.time()
    n_new = 0
    fh = args.out.open("a", encoding="utf-8")
    try:
        for rec in sorted(cls.values(), key=lambda r: (int(r["pair"]), int(r["seed"]))):
            pair, seed, label = int(rec["pair"]), int(rec["seed"]), rec["label"]
            ref = refs(pair, seed)
            tasks = [("native", args.census_dir / f"pair{pair:03d}_seed{seed}_SS.jpg")]
            if label == "ME":
                tasks.append(("routed", args.repair_dir / f"lsda_me_p{pair:03d}_s{seed}.jpg"))
                tasks.append(("uniform", args.repair_dir / f"lsda_me_p{pair:03d}_s{seed}.jpg"))
            elif label == "KA":
                tasks.append(("routed", args.repair_dir / f"lsda_ka_p{pair:03d}_s{seed}.jpg"))
                tasks.append(("uniform", args.repair_dir / f"lsda_uniform_p{pair:03d}_s{seed}.jpg"))
            for method, path in tasks:
                rid = f"{method}_p{pair:03d}_s{seed}"
                if rid in done or not path.exists():
                    continue
                img = Image.open(path).convert("RGB")
                w, h = img.size
                left_m, right_m = half_masks(w, h)
                row = {"id": rid, "pair": pair, "seed": seed, "label": label, "method": method}
                for tag, m in (("L", left_m), ("R", right_m)):
                    tok = patch_tokens(model, proc, img, m)
                    row[f"siglip_{tag}_A"] = masked_maxcos(ref["a"], tok)
                    row[f"siglip_{tag}_B"] = masked_maxcos(ref["b"], tok)
                row["attr_siglip"] = "correct" if row["siglip_L_A"] > row["siglip_L_B"] and row["siglip_R_B"] > row["siglip_R_A"] else "drift"
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                done.add(rid)
                n_new += 1
                if n_new % 20 == 0:
                    print(f"siglip: {n_new} done, {time.time()-t0:.0f}s", flush=True)
    finally:
        fh.close()
    print(f"saved {n_new} rows (siglip) in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
