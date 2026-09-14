"""Objective (non-VLM) binding metrics for the KA/ME census — memory-light, single process.

Two passes, one model resident at a time:
  --model clip : CLIP image/text region-vs-reference/text attributions
  --model dino : DINOv2 CLS region-vs-reference attributions
Rows are written incrementally and skipped on re-run.
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

import torch
from PIL import Image

ROOT = Path(r"D:\Python\MMDIT")
MODELS = ROOT / "models"
DEV = "cpu"
torch.set_num_threads(max(1, (os.cpu_count() or 4)))

sys.path.insert(0, str(ROOT / "code" / "mmdit_causal"))
from pairs100 import a_only_prompt, b_only_prompt  # noqa: E402


def _name(repo: str) -> str:
    return repo.replace("/", "__")


def load(model_kind: str):
    if model_kind == "clip":
        from transformers import CLIPModel, CLIPProcessor
        name = _name("openai/clip-vit-large-patch14")
        model = CLIPModel.from_pretrained(str(MODELS / name), torch_dtype=torch.float32).to(DEV).eval()
        proc = CLIPProcessor.from_pretrained(str(MODELS / name))
        return model, proc
    from transformers import AutoImageProcessor, AutoModel
    name = _name("facebook/dinov2-large")
    model = AutoModel.from_pretrained(str(MODELS / name), torch_dtype=torch.float32).to(DEV).eval()
    proc = AutoImageProcessor.from_pretrained(str(MODELS / name))
    return model, proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["clip", "dino"], required=True)
    ap.add_argument("--classification", type=Path, required=True)
    ap.add_argument("--census-dir", type=Path, required=True)
    ap.add_argument("--repair-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cls = json.loads(args.classification.read_text(encoding="utf-8"))
    model, proc = load(args.model)
    print(f"model={args.model} device={DEV}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])

    @torch.no_grad()
    def embed_image(pil):
        if args.model == "clip":
            out = model.get_image_features(**proc(images=pil, return_tensors="pt").to(DEV))
            return out.pooler_output if hasattr(out, "pooler_output") else out
        return model(**proc(images=pil, return_tensors="pt").to(DEV)).last_hidden_state[:, 0]

    @torch.no_grad()
    def embed_text(text):
        out = model.get_text_features(**proc(text=[text], return_tensors="pt", padding=True).to(DEV))
        return out.pooler_output if hasattr(out, "pooler_output") else out

    def cos(a, b):
        a = a / a.norm(dim=-1, keepdim=True)
        b = b / b.norm(dim=-1, keepdim=True)
        return float((a @ b.T).item())

    ref_cache: dict = {}

    def refs(pair, seed):
        key = (pair, seed)
        if key in ref_cache:
            return ref_cache[key]
        a = Image.open(args.census_dir / f"pair{pair:03d}_seed{seed}_A.jpg").convert("RGB")
        b = Image.open(args.census_dir / f"pair{pair:03d}_seed{seed}_B.jpg").convert("RGB")
        out = {"a": embed_image(a), "b": embed_image(b)}
        if args.model == "clip":
            out["txt_a"] = embed_text(a_only_prompt(pair))
            out["txt_b"] = embed_text(b_only_prompt(pair))
        ref_cache[key] = out
        return out

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
                row = {"id": rid, "pair": pair, "seed": seed, "label": label, "method": method}
                for tag, crop in (("L", img.crop((0, 0, w // 2, h))), ("R", img.crop((w // 2, 0, w, h)))):
                    ei = embed_image(crop)
                    if args.model == "clip":
                        row[f"clip_i_{tag}_A"] = cos(ei, ref["a"])
                        row[f"clip_i_{tag}_B"] = cos(ei, ref["b"])
                        row[f"clip_t_{tag}_A"] = cos(ei, ref["txt_a"])
                        row[f"clip_t_{tag}_B"] = cos(ei, ref["txt_b"])
                    else:
                        row[f"dino_{tag}_A"] = cos(ei, ref["a"])
                        row[f"dino_{tag}_B"] = cos(ei, ref["b"])
                if args.model == "clip":
                    row["attr_clip_i"] = "correct" if row["clip_i_L_A"] > row["clip_i_L_B"] and row["clip_i_R_B"] > row["clip_i_R_A"] else "drift"
                    row["attr_clip_t"] = "correct" if row["clip_t_L_A"] > row["clip_t_L_B"] and row["clip_t_R_B"] > row["clip_t_R_A"] else "drift"
                else:
                    row["attr_dino"] = "correct" if row["dino_L_A"] > row["dino_L_B"] and row["dino_R_B"] > row["dino_R_A"] else "drift"
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                done.add(rid)
                n_new += 1
                if n_new % 25 == 0:
                    print(f"{args.model}: {n_new} done, {time.time()-t0:.0f}s", flush=True)
    finally:
        fh.close()
    print(f"saved {n_new} rows ({args.model}) in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
