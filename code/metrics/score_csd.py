"""CSD (Content-Style Decomposition) style-embedding binding scores.

Loads ONLY the CSD model (CLIP ViT-L/14 backbone + style/content projections),
computes style-embedding cosine between candidate entity regions and A/B-alone
references, and writes csd_scores.jsonl (merged into the analysis by eval_id).

Needs:
  - references/art_arena/CSD (model.py / utils.py from the ArtArena repo)
  - models/csd_checkpoint.pt (Google Drive checkpoint)
  - OpenAI CLIP package at code/metrics/vendor/CLIP-main
"""

from __future__ import annotations

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

sys.path.insert(0, str(Path(r"D:\Python\MMDIT\code\metrics\vendor\CLIP-main")))
sys.path.insert(0, str(Path(r"D:\Python\MMDIT\references\art_arena")))  # CSD package

ROOT = Path(r"D:\Python\MMDIT")
EXP = ROOT / "experiment" / "2026_8_31_EXP_2"
MODELS = ROOT / "models"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

from torchvision import transforms  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402


def load_csd():
    from CSD.model import CSD_CLIP
    from CSD.utils import convert_state_dict

    ckpt = MODELS / "csd_checkpoint.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"CSD checkpoint missing: {ckpt}")
    model = CSD_CLIP("vit_large", "default")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(convert_state_dict(sd), strict=False)
    model = model.to(DEV).eval()
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    pre = transforms.Compose([
        transforms.Resize(size=224, interpolation=TF.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return model, pre


@torch.no_grad()
def style_emb(model, pre, pil: Image.Image) -> torch.Tensor:
    x = pre(pil.convert("RGB")).unsqueeze(0).to(DEV)
    _, _, style = model(x)
    return style[0].cpu()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a @ b.T).item())


def main() -> None:
    model, pre = load_csd()
    print(f"CSD loaded on {DEV}", flush=True)

    index = json.loads((EXP / "manifests" / "eval_index.json").read_text(encoding="utf-8"))
    SEG = EXP / "segmentation"
    mask_root = {
        "native_SS": SEG / "masks_native_SS",
        "lsda_clean": SEG / "masks_lsda_clean",
    }
    out = EXP / "ratings" / "automated" / "csd_scores.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["eval_id"])
            except json.JSONDecodeError:
                pass

    ref_cache: dict[str, torch.Tensor] = {}

    def ref_emb(sid: str, which: str, pil: Image.Image):
        key = f"{sid}:{which}"
        if key not in ref_cache:
            ref_cache[key] = style_emb(model, pre, pil)
        return ref_cache[key]

    t0 = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(index, 1):
            eid = r["eval_id"]
            if eid in done:
                continue
            cond = r["condition"]
            sid = r["sample_id"]
            try:
                cand = Image.open(r["images"][cond]).convert("RGB")
                a_img = Image.open(r["images"]["standalone_A"]).convert("RGB")
                b_img = Image.open(r["images"]["standalone_B"]).convert("RGB")
                m = np.load(mask_root[cond] / f"{sid}_mask.npz")
                left_m, right_m = m["left"], m["right"]
                h, w = left_m.shape
                if float(left_m.mean()) < 0.02:
                    left_m = np.zeros((h, w), np.uint8)
                    left_m[:, : w // 2] = 1
                if float(right_m.mean()) < 0.02:
                    right_m = np.zeros((h, w), np.uint8)
                    right_m[:, w // 2 :] = 1
                ys, xs = np.where(left_m > 0)
                lc = cand.crop((xs.min(), ys.min(), xs.max(), ys.max())) if len(xs) else cand
                ys, xs = np.where(right_m > 0)
                rc = cand.crop((xs.min(), ys.min(), xs.max(), ys.max())) if len(xs) else cand
                le = style_emb(model, pre, lc)
                re_ = style_emb(model, pre, rc)
                eA = ref_emb(sid, "A", a_img)
                eB = ref_emb(sid, "B", b_img)
                rec = {
                    "eval_id": eid,
                    "csd_L_A": cos(le, eA),
                    "csd_L_B": cos(le, eB),
                    "csd_R_A": cos(re_, eA),
                    "csd_R_B": cos(re_, eB),
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if i % 100 == 0:
                    print(f"[{i}/{len(index)}] {time.time()-t0:.0f}s", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR {eid}: {type(exc).__name__}: {exc}", flush=True)
    print(f"CSD done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
