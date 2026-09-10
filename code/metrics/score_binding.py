"""Automated binding scores for SS/LSDA images (fully local, no VLM API).

Metrics per entity (left/right) of each candidate image:
  - siglip_cp: MaSC-style masked-maxcos of SigLIP2 patch tokens between the
    reference (A-alone / B-alone foreground) and the candidate entity region
  - clip_i:   CLIP image-embedding cosine between region crop and reference
  - clip_t:   CLIP region-crop vs entity short-prompt text cosine
  - dino:     DINOv2 [CLS] cosine between region crop and reference

Attribution per side: pick A vs B by larger score; correct binding =
left->A and right->B. All continuous scores are saved.

Usage:
  python score_binding.py [--limit N] [--skip-blip] [--condition native_SS,lsda_clean]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Force offline: all model weights are local; never touch the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image

if not torch.cuda.is_available():
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    print(f"CPU mode: using {torch.get_num_threads()} threads", flush=True)

ROOT = Path(r"D:\Python\MMDIT")
EXP = ROOT / "experiment" / "2026_8_31_EXP_2"
MODELS = ROOT / "models"
OUT = EXP / "ratings" / "automated"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sig_name(repo: str) -> str:
    return repo.replace("/", "__")


def load_siglip2():
    from transformers import AutoProcessor, Siglip2VisionModel

    repo = "google/siglip2-so400m-patch16-naflex"
    name = sig_name(repo)
    dt = torch.float16 if DEV == "cuda" else torch.float32
    model = Siglip2VisionModel.from_pretrained(str(MODELS / name), torch_dtype=dt).to(DEV).eval()
    proc = AutoProcessor.from_pretrained(str(MODELS / name))
    return model, proc


def load_clip():
    from transformers import CLIPModel, CLIPProcessor

    name = sig_name("openai/clip-vit-large-patch14")
    dt = torch.float16 if DEV == "cuda" else torch.float32
    model = CLIPModel.from_pretrained(str(MODELS / name), torch_dtype=dt).to(DEV).eval()
    proc = CLIPProcessor.from_pretrained(str(MODELS / name))
    return model, proc


def load_dino():
    from transformers import AutoModel, AutoImageProcessor

    name = sig_name("facebook/dinov2-large")
    dt = torch.float16 if DEV == "cuda" else torch.float32
    model = AutoModel.from_pretrained(str(MODELS / name), torch_dtype=dt).to(DEV).eval()
    proc = AutoImageProcessor.from_pretrained(str(MODELS / name))
    return model, proc


def load_blip():
    from transformers import BlipForQuestionAnswering, BlipProcessor

    name = sig_name("Salesforce/blip-vqa-base")
    dt = torch.float16 if DEV == "cuda" else torch.float32
    proc = BlipProcessor.from_pretrained(str(MODELS / name))
    model = BlipForQuestionAnswering.from_pretrained(str(MODELS / name), torch_dtype=dt).to(DEV).eval()
    return model, proc


@torch.no_grad()
def siglip_patch_tokens(model, proc, pil_img: Image.Image, mask: np.ndarray | None):
    """Return (B, D) patch-token matrix (optionally masked) for an image.

    Input is resized to a fixed square (224) so the patch grid is well-defined
    even for SigLIP2 NaFlex (variable-aspect) models.
    """
    S = 224
    pil_img = pil_img.resize((S, S), Image.BILINEAR)
    inputs = proc(images=pil_img, return_tensors="pt", size={"height": S, "width": S}).to(DEV)
    # NaFlex processor emits patch-embedded inputs (pixel_values, pixel_attention_mask,
    # spatial_shapes); Siglip2VisionModel consumes them directly.
    out = model(
        pixel_values=inputs["pixel_values"],
        pixel_attention_mask=inputs["pixel_attention_mask"],
        spatial_shapes=inputs["spatial_shapes"],
    )
    feats = out.last_hidden_state[0]  # [N, D] — NaFlex vision output IS the patch tokens (no CLS)
    tokens = feats
    tokens = tokens / tokens.norm(dim=-1, keepdim=True)
    if mask is not None:
        gh, gw = inputs["spatial_shapes"][0].tolist()
        grid = int(round((gh * gw) ** 0.5))
        m = Image.fromarray((mask * 255).astype(np.uint8)).resize((grid, grid), Image.NEAREST)
        m = np.asarray(m) > 127
        tokens = tokens[m.reshape(-1)]
    return tokens.cpu()


def masked_maxcos(ref_tokens: torch.Tensor, cand_tokens: torch.Tensor) -> float:
    """MaSC CP: mean over reference patches of max cosine to candidate patches."""
    if ref_tokens.shape[0] == 0 or cand_tokens.shape[0] == 0:
        return float("nan")
    dev = ref_tokens.device
    if dev.type == "cpu" and torch.cuda.is_available():
        ref_tokens = ref_tokens.cuda()
        cand_tokens = cand_tokens.cuda()
    sims = ref_tokens @ cand_tokens.T  # [R, C]
    out = float(sims.max(dim=1).values.float().mean())
    return out


@torch.no_grad()
def embed_clip_image(model, proc, pil_img: Image.Image) -> torch.Tensor:
    inputs = proc(images=pil_img, return_tensors="pt").to(DEV)
    out = model.get_image_features(**inputs)
    emb = out.pooler_output if hasattr(out, "pooler_output") else out
    return emb.cpu()


@torch.no_grad()
def embed_clip_text(model, proc, text: str) -> torch.Tensor:
    inputs = proc(text=[text], return_tensors="pt", padding=True).to(DEV)
    out = model.get_text_features(**inputs)
    emb = out.pooler_output if hasattr(out, "pooler_output") else out
    return emb.cpu()


@torch.no_grad()
def embed_dino(model, proc, pil_img: Image.Image) -> torch.Tensor:
    inputs = proc(images=pil_img, return_tensors="pt").to(DEV)
    return model(**inputs).last_hidden_state[:, 0].cpu()


def crop_by_mask(pil_img: Image.Image, mask: np.ndarray, margin: float = 0.04) -> Image.Image:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return pil_img
    h, w = mask.shape
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    dx, dy = int((x1 - x0) * margin), int((y1 - y0) * margin)
    x0, x1 = max(0, x0 - dx), min(w - 1, x1 + dx)
    y0, y1 = max(0, y0 - dy), min(h - 1, y1 + dy)
    return pil_img.crop((x0, y0, x1, y1))


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return float((a @ b.T).item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-blip", action="store_true")
    ap.add_argument("--condition", default="native_SS,lsda_clean")
    args = ap.parse_args()

    index = json.loads((EXP / "manifests" / "eval_index.json").read_text(encoding="utf-8"))
    conds = set(args.condition.split(","))
    rows = [r for r in index if r["condition"] in conds]
    if args.limit:
        rows = rows[: args.limit]

    print(f"device: {DEV}; images to score: {len(rows)}", flush=True)
    siglip_model, siglip_proc = load_siglip2()
    clip_model, clip_proc = load_clip()
    dino_model, dino_proc = load_dino()
    blip_model = blip_proc = None
    if not args.skip_blip:
        blip_model, blip_proc = load_blip()

    SEG = EXP / "segmentation"
    mask_root = {
        "native_SS": SEG / "masks_native_SS",
        "lsda_clean": SEG / "masks_lsda_clean",
        "standalone_A": SEG / "masks_standalone_A",
        "standalone_B": SEG / "masks_standalone_B",
    }

    ref_cache: dict[str, dict] = {}

    def ref_features(sample_id: str, cond_ref: str, pil: Image.Image, mask: np.ndarray):
        """SigLIP fg tokens + CLIP embed + DINO embed of a reference."""
        key = f"{sample_id}:{cond_ref}"
        if key in ref_cache:
            return ref_cache[key]
        pil = pil.convert("RGB")
        out = {
            "siglip": siglip_patch_tokens(siglip_model, siglip_proc, pil, mask),
            "clip": embed_clip_image(clip_model, clip_proc, pil),
            "dino": embed_dino(dino_model, dino_proc, pil),
        }
        ref_cache[key] = out
        return out

    OUT.mkdir(parents=True, exist_ok=True)
    out_jsonl = OUT / "binding_scores.jsonl"
    done = set()
    if out_jsonl.exists():
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["eval_id"])
            except json.JSONDecodeError:
                pass

    t0 = time.time()
    with out_jsonl.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(rows, 1):
            eid = r["eval_id"]
            if eid in done:
                print(f"skip done {eid}", flush=True)
                continue
            sid = r["sample_id"]
            try:
                cand_img = Image.open(r["images"][r["condition"]]).convert("RGB")
                a_img = Image.open(r["images"]["standalone_A"]).convert("RGB")
                b_img = Image.open(r["images"]["standalone_B"]).convert("RGB")
                m = np.load(mask_root[r["condition"]] / f"{sid}_mask.npz")
                left_m, right_m = m["left"], m["right"]
                a_m = np.load(mask_root["standalone_A"] / f"{sid}_mask.npz")["fg"]
                b_m = np.load(mask_root["standalone_B"] / f"{sid}_mask.npz")["fg"]

                # fallback: empty/tiny side mask -> use that image half, flagged
                rec0 = {"eval_id": eid, "mask_fallback_left": 0, "mask_fallback_right": 0}
                h, w = left_m.shape
                if float(left_m.mean()) < 0.02:
                    left_m = np.zeros((h, w), np.uint8)
                    left_m[:, : w // 2] = 1
                    rec0["mask_fallback_left"] = 1
                if float(right_m.mean()) < 0.02:
                    right_m = np.zeros((h, w), np.uint8)
                    right_m[:, w // 2 :] = 1
                    rec0["mask_fallback_right"] = 1

                left_tok = siglip_patch_tokens(siglip_model, siglip_proc, cand_img, left_m)
                right_tok = siglip_patch_tokens(siglip_model, siglip_proc, cand_img, right_m)
                refA = ref_features(sid, "A", a_img, a_m)
                refB = ref_features(sid, "B", b_img, b_m)

                left_crop = crop_by_mask(cand_img, left_m)
                right_crop = crop_by_mask(cand_img, right_m)
                clip_L = embed_clip_image(clip_model, clip_proc, left_crop)
                clip_R = embed_clip_image(clip_model, clip_proc, right_crop)
                dino_L = embed_dino(dino_model, dino_proc, left_crop)
                dino_R = embed_dino(dino_model, dino_proc, right_crop)
                clip_tA = embed_clip_text(clip_model, clip_proc, r["entity_A"])
                clip_tB = embed_clip_text(clip_model, clip_proc, r["entity_B"])

                rec = {
                    "eval_id": eid,
                    "sample_id": sid,
                    "pair_id": r["pair_id"],
                    "seed_group": r["seed_group"],
                    "replicate": r["replicate"],
                    "latent_seed": r["latent_seed"],
                    "condition": r["condition"],
                    "mask_left_area": float(left_m.mean()),
                    "mask_right_area": float(right_m.mean()),
                    "mask_fallback_left": rec0["mask_fallback_left"],
                    "mask_fallback_right": rec0["mask_fallback_right"],
                }
                # siglip masked-maxcos CP
                cp = {
                    "L_A": masked_maxcos(refA["siglip"], left_tok),
                    "L_B": masked_maxcos(refB["siglip"], left_tok),
                    "R_A": masked_maxcos(refA["siglip"], right_tok),
                    "R_B": masked_maxcos(refB["siglip"], right_tok),
                }
                rec.update({f"siglip_cp_{k}": v for k, v in cp.items()})
                # clip image
                ci = {
                    "L_A": cos(clip_L, refA["clip"]),
                    "L_B": cos(clip_L, refB["clip"]),
                    "R_A": cos(clip_R, refA["clip"]),
                    "R_B": cos(clip_R, refB["clip"]),
                }
                rec.update({f"clip_i_{k}": v for k, v in ci.items()})
                # clip text
                ct = {
                    "L_A": cos(clip_L, clip_tA),
                    "L_B": cos(clip_L, clip_tB),
                    "R_A": cos(clip_R, clip_tA),
                    "R_B": cos(clip_R, clip_tB),
                }
                rec.update({f"clip_t_{k}": v for k, v in ct.items()})
                # dino
                dn = {
                    "L_A": cos(dino_L, refA["dino"]),
                    "L_B": cos(dino_L, refB["dino"]),
                    "R_A": cos(dino_R, refA["dino"]),
                    "R_B": cos(dino_R, refB["dino"]),
                }
                rec.update({f"dino_{k}": v for k, v in dn.items()})

                # attributions
                rec["attr_siglip"] = "correct" if cp["L_A"] > cp["L_B"] and cp["R_B"] > cp["R_A"] else "drift"
                rec["attr_clip_i"] = "correct" if ci["L_A"] > ci["L_B"] and ci["R_B"] > ci["R_A"] else "drift"
                rec["attr_clip_t"] = "correct" if ct["L_A"] > ct["L_B"] and ct["R_B"] > ct["R_A"] else "drift"
                rec["attr_dino"] = "correct" if dn["L_A"] > dn["L_B"] and dn["R_B"] > dn["R_A"] else "drift"

                if blip_model is not None:
                    qA = f"Is this {r['entity_A']}?"
                    qB = f"Is this {r['entity_B']}?"
                    bl = {}
                    for side, crop in (("L", left_crop), ("R", right_crop)):
                        for tag, q in (("A", qA), ("B", qB)):
                            inp = blip_proc(images=crop, text=q, return_tensors="pt").to(DEV)
                            with torch.no_grad():
                                out_ids = blip_model.generate(**inp, max_new_tokens=3, do_sample=False)
                            ans = blip_proc.decode(out_ids[0], skip_special_tokens=True).strip().lower()
                            bl[f"{side}_{tag}"] = ans
                    rec["blip"] = bl
                    rec["attr_blip"] = (
                        "correct"
                        if bl["L_A"].startswith("yes") and not bl["L_B"].startswith("yes")
                        and bl["R_B"].startswith("yes") and not bl["R_A"].startswith("yes")
                        else "drift"
                    )

                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"[{i}/{len(rows)}] {eid} {r['condition']} cp_L_A={cp['L_A']:.3f} cp_L_B={cp['L_B']:.3f} "
                      f"cp_R_A={cp['R_A']:.3f} cp_R_B={cp['R_B']:.3f} attr_siglip={rec['attr_siglip']}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR {eid}: {type(exc).__name__}: {exc}", flush=True)
    print(f"done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
