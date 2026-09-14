"""DINOv2 metrics: (a) Gram style distance, (b) enhanced-prototype k-NN.

Both use frozen DINOv2-large (no training). Prototypes = deterministic augmentations of the
standalone A/B references. Regions = left/right halves. Outputs per-arm correctness for each.
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

sys.path.insert(0, str(Path(__file__).parent))
from arms import load_classification, refs_for, resolve  # noqa: E402

MODELS = Path(r"D:\Python\MMDIT\models")
DEV = "cpu"
torch.set_num_threads(max(1, (os.cpu_count() or 4)))
BATCH = 8


def augmentations(pil: Image.Image):
    w, h = pil.size
    def crop(frac, cx, cy):
        bw, bh = int(w * frac), int(h * frac)
        x0 = int(cx * w - bw / 2)
        y0 = int(cy * h - bh / 2)
        x0 = max(0, min(w - bw, x0))
        y0 = max(0, min(h - bh, y0))
        return pil.crop((x0, y0, x0 + bw, y0 + bh))
    return [
        pil,
        pil.transpose(Image.FLIP_LEFT_RIGHT),
        crop(0.8, 0.5, 0.5),
        crop(0.75, 0.3, 0.3),
        crop(0.75, 0.7, 0.7),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", type=Path, required=True)
    ap.add_argument("--census-dir", type=Path, required=True)
    ap.add_argument("--lsda-dir", type=Path, required=True)
    ap.add_argument("--binding-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--arms", default="SS,LSDA,Binding")
    args = ap.parse_args()
    wanted = {a.strip() for a in args.arms.split(",") if a.strip()}
    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])

    from transformers import AutoImageProcessor, AutoModel
    name = "facebook__dinov2-large"
    model = AutoModel.from_pretrained(str(MODELS / name), torch_dtype=torch.float32).to(DEV).eval()
    proc = AutoImageProcessor.from_pretrained(str(MODELS / name))
    print("dino loaded", flush=True)

    @torch.no_grad()
    def forward(pils):
        inputs = proc(images=pils, return_tensors="pt").to(DEV)
        out = model(**inputs).last_hidden_state  # [B, 1+N, D]
        cls = out[:, 0]  # [B, D]
        patches = out[:, 1:]  # [B, N, D]
        return cls / cls.norm(dim=-1, keepdim=True), patches

    def gram(patches):
        p = patches[0]  # [N, D]
        p = p / (p.norm(dim=-1, keepdim=True) + 1e-8)
        g = p.T @ p / p.shape[0]
        return g

    def gram_dist(patches_region, patches_ref):
        g1, g2 = gram(patches_region), gram(patches_ref)
        return float(torch.norm(g1 - g2))

    rows = load_classification(args.classification)
    args.out.parent.mkdir(parents=True, exist_ok=True)
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
            A, B = Image.open(a_ref).convert("RGB"), Image.open(b_ref).convert("RGB")
            a_augs, b_augs = augmentations(A), augmentations(B)
            cls_a, patch_a = forward(a_augs)
            cls_b, patch_b = forward(b_augs)
            proto = torch.cat([cls_a, cls_b], 0)  # [Ka+Kb, D]
            labels = np.array([0] * cls_a.shape[0] + [1] * cls_b.shape[0])

            arms = resolve(rec, args.census_dir, args.lsda_dir, args.binding_dir)
            for arm, path in arms.items():
                rid = f"{arm}_p{rec['pair']:03d}_s{rec['seed']}"
                if arm not in wanted or rid in done or not path.exists():
                    continue
                img = Image.open(path).convert("RGB")
                w, h = img.size
                regions = [img.crop((0, 0, w // 2, h)), img.crop((w // 2, 0, w, h))]
                cls_r, patch_r = forward(regions)
                sims = cls_r @ proto.T  # [2, K]
                knn_ok = []
                for side in range(2):
                    top = torch.topk(sims[side], k=args.knn_k).indices.cpu().numpy()
                    vote = int(round(labels[top].mean()))
                    knn_ok.append((side == 0 and vote == 0) or (side == 1 and vote == 1))
                knn_correct = all(knn_ok)
                d = {
                    "L_A": gram_dist(patch_r[0:1], patch_a[0:1]), "L_B": gram_dist(patch_r[0:1], patch_b[0:1]),
                    "R_A": gram_dist(patch_r[1:2], patch_a[0:1]), "R_B": gram_dist(patch_r[1:2], patch_b[0:1]),
                }
                gram_correct = (d["L_A"] < d["L_B"]) and (d["R_B"] < d["R_A"])
                fh.write(json.dumps({
                    "id": rid, "arm": arm,
                    "pair": int(rec["pair"]), "seed": int(rec["seed"]), "label": rec["label"],
                    "knn_correct": bool(knn_correct), "gram_correct": bool(gram_correct), **d,
                }) + "\n")
                fh.flush()
                done.add(rid)
                n_new += 1
                if n_new % 25 == 0:
                    print(f"dino: {n_new} rows, {time.time()-t0:.0f}s", flush=True)
    finally:
        fh.close()
    print(f"dino rows={n_new} in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
