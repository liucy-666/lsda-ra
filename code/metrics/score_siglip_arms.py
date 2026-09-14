"""SigLIP2 MaSC-CP binding metric for the three census arms."""
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

sys.path.insert(0, str(Path(__file__).parent))
from arms import load_classification, refs_for, resolve  # noqa: E402

MODELS = Path(r"D:\Python\MMDIT\models")
DEV = "cpu"
torch.set_num_threads(max(1, (os.cpu_count() or 4)))


def load(dtype_name: str = "bfloat16"):
    from transformers import AutoProcessor, Siglip2VisionModel
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}[dtype_name]
    name = "google__siglip2-so400m-patch16-naflex"
    model = Siglip2VisionModel.from_pretrained(str(MODELS / name), torch_dtype=dtype).to(DEV).eval()
    proc = AutoProcessor.from_pretrained(str(MODELS / name))
    return model, proc, dtype


@torch.no_grad()
def patch_tokens(model, proc, pil, mask=None, dtype=torch.bfloat16):
    s = 224
    img = pil.convert("RGB").resize((s, s), Image.BILINEAR) if mask is None else pil
    inputs = proc(images=img, return_tensors="pt", size={"height": s, "width": s}).to(DEV)
    pixel_values = inputs["pixel_values"].to(dtype)
    out = model(
        pixel_values=pixel_values,
        pixel_attention_mask=inputs["pixel_attention_mask"],
        spatial_shapes=inputs["spatial_shapes"],
    )
    tokens = out.last_hidden_state[0]
    tokens = tokens / tokens.norm(dim=-1, keepdim=True)
    if mask is not None:
        gh, gw = inputs["spatial_shapes"][0].tolist()
        grid = int(round((gh * gw) ** 0.5))
        m = Image.fromarray((mask * 255).astype(np.uint8)).resize((grid, grid), Image.NEAREST)
        sel = (np.asarray(m) > 127).reshape(-1)
        if sel.sum() > 0:
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
    ap.add_argument("--lsda-dir", type=Path, required=True)
    ap.add_argument("--binding-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"])
    ap.add_argument("--arms", default="SS,LSDA,Binding")
    args = ap.parse_args()
    wanted = {a.strip() for a in args.arms.split(",") if a.strip()}

    model, proc, dtype = load(args.dtype)
    print(f"siglip loaded ({args.dtype})", flush=True)
    rows = load_classification(args.classification)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    t0 = time.time()
    fh = args.out.open("a", encoding="utf-8")
    n_new = 0
    try:
        for rec in rows:
            if not any(f"{arm}_p{rec['pair']:03d}_s{rec['seed']}" not in done for arm in wanted):
                continue
            a_ref, b_ref = refs_for(rec, args.census_dir)
            if not (a_ref.exists() and b_ref.exists()):
                continue
            tok_a = patch_tokens(model, proc, Image.open(a_ref).convert("RGB"), dtype=dtype)
            tok_b = patch_tokens(model, proc, Image.open(b_ref).convert("RGB"), dtype=dtype)
            arms = resolve(rec, args.census_dir, args.lsda_dir, args.binding_dir)
            for arm, path in arms.items():
                rid = f"{arm}_p{rec['pair']:03d}_s{rec['seed']}"
                if arm not in wanted or rid in done or not path.exists():
                    continue
                img = Image.open(path).convert("RGB")
                w, h = img.size
                left = np.zeros((h, w), np.uint8)
                left[:, : w // 2] = 1
                right = np.zeros((h, w), np.uint8)
                right[:, w // 2 :] = 1
                lp = patch_tokens(model, proc, img, left, dtype=dtype)
                rp = patch_tokens(model, proc, img, right, dtype=dtype)
                d = {
                    "L_A": masked_maxcos(tok_a, lp), "L_B": masked_maxcos(tok_b, lp),
                    "R_A": masked_maxcos(tok_a, rp), "R_B": masked_maxcos(tok_b, rp),
                }
                correct = (d["L_A"] > d["L_B"]) and (d["R_B"] > d["R_A"])
                fh.write(json.dumps({
                    "id": rid, "arm": arm,
                    "pair": int(rec["pair"]), "seed": int(rec["seed"]), "label": rec["label"],
                    "correct": bool(correct), **d,
                }) + "\n")
                fh.flush()
                done.add(rid)
                n_new += 1
                if n_new % 20 == 0:
                    print(f"siglip: {n_new} rows, {time.time()-t0:.0f}s", flush=True)
    finally:
        fh.close()
    print(f"siglip rows={n_new} in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
