"""Diagnostics: ref swap check, margin distribution, 2x2 agreement vs VLM."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:\Python\MMDIT\code\metrics")
ROOT = Path(r"D:\Python\MMDIT")
EXP = ROOT / "experiment" / "2026_8_31_EXP_2"
MODELS = ROOT / "models"

# 1) A/B reference swap check via CLIP-T: standalone_A image vs entity_A text should win
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

dev = "cuda" if torch.cuda.is_available() else "cpu"
clip = CLIPModel.from_pretrained(str(MODELS / "openai__clip-vit-large-patch14"), torch_dtype=torch.float16).to(dev).eval()
proc = CLIPProcessor.from_pretrained(str(MODELS / "openai__clip-vit-large-patch14"))

index = json.loads((EXP / "manifests" / "eval_index.json").read_text(encoding="utf-8"))
# unique samples with entity texts
seen = {}
for r in index:
    seen.setdefault(r["sample_id"], r)

refs_a = refs_b = 0
refs_total = 0
for sid, r in list(seen.items())[:200]:
    for cond, path in (("A", r["images"]["standalone_A"]), ("B", r["images"]["standalone_B"])):
        img = Image.open(path).convert("RGB")
        inp = proc(images=img, text=[r["entity_A"], r["entity_B"]], return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            out = clip(**inp)
        simA = float((out.image_embeds * out.text_embeds[0]).sum(-1))
        simB = float((out.image_embeds * out.text_embeds[1]).sum(-1))
        refs_total += 1
        if cond == "A" and simA > simB:
            refs_a += 1
        if cond == "B" and simB > simA:
            refs_b += 1
print(f"[ref check] n={refs_total}  standalone_A matches entity_A text: {refs_a}/200  standalone_B matches entity_B text: {refs_b}/200")

# 2) margin distribution from binding scores
scores = [json.loads(l) for l in (EXP / "ratings" / "automated" / "binding_scores.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
margins = []
for rec in scores:
    for pre in ("siglip_cp", "clip_i", "clip_t", "dino"):
        if f"{pre}_L_A" in rec:
            dL = abs(rec[f"{pre}_L_A"] - rec[f"{pre}_L_B"])
            dR = abs(rec[f"{pre}_R_A"] - rec[f"{pre}_R_B"])
            margins.append(dL)
            margins.append(dR)
m = np.array(margins)
print(f"[margin] n={len(m)}  mean={m.mean():.4f} median={np.median(m):.4f} p10={np.percentile(m,10):.4f} p90={np.percentile(m,90):.4f} frac<0.05={float((m<0.05).mean()):.2%}")

# 3) 2x2 agreement: consensus vs dual-VLM (native_SS only)
# recompute consensus without blip
METHODS = ["siglip", "clip_i", "clip_t", "dino"]

def side_attr(rec, method):
    pre = {"siglip": "siglip_cp", "clip_i": "clip_i", "clip_t": "clip_t", "dino": "dino"}[method]
    l = "A" if rec[f"{pre}_L_A"] >= rec[f"{pre}_L_B"] else "B"
    r = "A" if rec[f"{pre}_R_A"] >= rec[f"{pre}_R_B"] else "B"
    return l, r

def consensus_correct(rec):
    votes = Counter()
    for m in METHODS:
        l, r = side_attr(rec, m)
        votes[f"L_{l}"] += 1
        votes[f"R_{r}"] += 1
    return (votes["L_A"] > votes["L_B"]) and (votes["R_B"] > votes["R_A"])

qwen = {}
VQA = ROOT / "experiment" / "2026_8_25_EXP_1" / "cultural100_records" / "experiment_4500" / "binary_vqa_v2"
for p in (VQA / "ratings" / "QWEN").glob("ratings_*.jsonl"):
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        qwen.setdefault(r["eval_id"], r)
gemini = {}
for p in (VQA / "ratings" / "GEMINI").glob("ratings_*.jsonl"):
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        gemini.setdefault(r["eval_id"], r)

tbl = Counter()
for rec in scores:
    eid = rec["eval_id"]
    if rec["condition"] != "native_SS":
        continue
    auto = consensus_correct(rec)
    q = qwen.get(eid, {}).get("correct_binding")
    g = gemini.get(eid, {}).get("correct_binding")
    if q is None or g is None:
        continue
    dual = bool(q) and bool(g)
    tbl[(auto, dual)] += 1
print("[2x2 native_SS] rows=auto-correct, cols=dual-VLM-correct:")
print(f"  auto_correct & dual_correct  : {tbl[(True, True)]}")
print(f"  auto_correct & dual_FALSE    : {tbl[(True, False)]}")
print(f"  auto_FALSE  & dual_correct   : {tbl[(False, True)]}")
print(f"  auto_FALSE  & dual_FALSE     : {tbl[(False, False)]}")
