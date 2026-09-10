"""Recompute instance masks for all evaluation images with SAM vit-base.

For each candidate image (native SS / LSDA) we prompt SAM at the expected
left/right entity positions (mini-grid of points) and keep one mask per side.
For standalone A/B images we prompt at the image center.

Outputs (per image):
  - experiment/2026_8_31_EXP_2/segmentation/masks_<cond>/<sample_id>_mask.npz
    containing uint8 arrays: left, right, fg (union), bg (complement)
  - a QC overlay PNG per image

Usage:
  python sam_masks.py --limit N        # pilot subset first
  python sam_masks.py --condition native_SS,lsda_clean,standalone_A,standalone_B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Force offline: all model weights are local; never touch the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image, ImageDraw

if not torch.cuda.is_available():
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    print(f"CPU mode: using {torch.get_num_threads()} threads", flush=True)

ROOT = Path(r"D:\Python\MMDIT")
EXP = ROOT / "experiment" / "2026_8_31_EXP_2"
SEG_ROOT = EXP / "segmentation"
MODEL_DIR = ROOT / "models" / "facebook__sam-vit-base"

POINTS_PER_SIDE = 3  # 3x3 mini-grid


def grid_around(cx: float, cy: float, img_w: int, img_h: int, r: float = 0.08) -> list[list[float]]:
    pts = []
    for dx in (-r, 0.0, r):
        for dy in (-r, 0.0, r):
            pts.append([(cx + dx) * img_w, (cy + dy) * img_h])
    return pts


def load_sam(model_dir: Path):
    """Load SAM via transformers SamModel (facebook/sam-vit-base is HF-format)."""
    import torch
    from transformers import SamModel, SamProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SAM model dir: {model_dir}, device: {dev}", flush=True)
    model = SamModel.from_pretrained(str(model_dir)).to(dev).eval()
    proc = SamProcessor.from_pretrained(str(model_dir))
    return model, proc, dev


def predict_mask(sam_model, sam_proc, image: np.ndarray, points_list: list[list[list[float]]], dev: str, prefer_side: str | None = None):
    """Return (best_mask, best_score).

    points_list: list of candidate point-sets; each set is a list of [x, y]
    points. All sets are fed as separate SAM "objects" in ONE forward pass
    (num_objects), then the best candidate (largest on-side mask) is chosen.
    prefer_side: 'left'|'right'|None.
    """
    import torch

    pil = Image.fromarray(image)
    inputs = sam_proc(
        images=[pil],
        input_points=[points_list],
        input_labels=[[[1] * len(pts) for pts in points_list]],
        return_tensors="pt",
    ).to(dev)
    with torch.no_grad():
        outputs = sam_model(**inputs, multimask_output=True)
    # pred_masks: [B=1, num_objects, num_masks, 256, 256]
    all_masks = sam_proc.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"],
        binarize=True,
    )[0].cpu().numpy().astype(np.uint8)  # [num_objects, num_masks, H, W]
    scores = outputs.iou_scores[0].cpu().numpy()  # [num_objects, num_masks]

    h_img, w_img = image.shape[:2]
    best_mask, best_score, best_key = None, None, None
    for oi in range(all_masks.shape[0]):
        for mi in range(all_masks.shape[1]):
            m = all_masks[oi, mi]
            ys, xs = np.where(m > 0)
            if len(xs) == 0:
                continue
            cx = xs.mean() / w_img
            area_frac = len(xs) / (h_img * w_img)
            on_side = (
                (prefer_side == "left" and cx < 0.5)
                or (prefer_side == "right" and cx >= 0.5)
                or prefer_side is None
            )
            if not on_side:
                continue
            if not (0.02 <= area_frac <= 0.70):  # reject whole-scene / tiny masks
                continue
            key = (float(scores[oi, mi]), area_frac)  # prefer high IoU, then bigger
            if best_key is None or key > best_key:
                best_key = key
                best_mask, best_score = m, float(scores[oi, mi])
    if best_mask is None:  # no suitable candidate -> largest overall
        areas = all_masks.reshape(all_masks.shape[0] * all_masks.shape[1], -1).sum(axis=1)
        idx = int(areas.argmax())
        oi, mi = divmod(idx, all_masks.shape[1])
        best_mask, best_score = all_masks[oi, mi], float(scores[oi, mi])
    if dev == "cuda":
        torch.cuda.empty_cache()
    return best_mask, best_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="pilot limit (0 = all)")
    ap.add_argument(
        "--condition",
        default="native_SS,lsda_clean,standalone_A,standalone_B",
    )
    args = ap.parse_args()

    index = json.loads(
        (EXP / "manifests" / "eval_index.json").read_text(encoding="utf-8")
    )
    # jobs: index conditions + standalone A/B (enumerated from image dirs)
    jobs: dict[str, list[dict]] = {}
    for r in index:
        cond = r["condition"]
        if cond not in args.condition.split(","):
            continue
        jobs.setdefault(cond, []).append({"sample_id": r["sample_id"], "img": r["images"][cond]})
    for cond in ("standalone_A", "standalone_B"):
        if cond not in args.condition.split(","):
            continue
        d = ROOT / "data" / {"standalone_A": "Standalone_A", "standalone_B": "Standalone_B"}[cond] / "2026_8_25_EXP_1"
        jobs[cond] = [
            {"sample_id": f.stem, "img": str(f)}
            for f in sorted(d.glob("*.jpg"))
        ]

    sam_model, sam_proc, dev = load_sam(MODEL_DIR)

    def side_point_sets(cx: float, w: int, h: int, n: int = 5) -> list[list[list[float]]]:
        """n candidate single-point sets spread around (cx, 0.5)."""
        xs = [cx + (i - (n - 1) / 2) * 0.06 for i in range(n)]
        return [[[x * w, 0.5 * h]] for x in xs]

    for cond, items in jobs.items():
        out_dir = SEG_ROOT / f"masks_{cond}"
        out_dir.mkdir(parents=True, exist_ok=True)
        items = sorted(items, key=lambda x: x["sample_id"])
        if args.limit:
            items = items[: args.limit]
        for i, it in enumerate(items, 1):
            sid = it["sample_id"]
            npz_path = out_dir / f"{sid}_mask.npz"
            if npz_path.exists():
                print(f"[{cond}] skip existing {sid}", flush=True)
                continue
            try:
                img = Image.open(it["img"]).convert("RGB")
                w, h = img.size
                arr = np.asarray(img)
                if cond in ("standalone_A", "standalone_B"):
                    m_fg, m_s = predict_mask(
                        sam_model, sam_proc, arr,
                        side_point_sets(0.5, w, h), dev, prefer_side=None,
                    )
                    left_m, left_s, right_m, right_s = m_fg, m_s, np.zeros_like(m_fg), m_s
                else:
                    left_m, left_s = predict_mask(
                        sam_model, sam_proc, arr,
                        side_point_sets(0.28, w, h), dev, prefer_side="left",
                    )
                    right_m, right_s = predict_mask(
                        sam_model, sam_proc, arr,
                        side_point_sets(0.72, w, h), dev, prefer_side="right",
                    )
                fg = (left_m | right_m).astype(np.uint8)
                bg = (1 - fg).astype(np.uint8)
                # overlap arbitration: left owns overlap (keeps one-hot)
                overlap = left_m & right_m
                right_m = right_m & ~overlap
                fg = (left_m | right_m).astype(np.uint8)
                bg = (1 - fg).astype(np.uint8)
                np.savez_compressed(
                    npz_path,
                    left=left_m.astype(np.uint8),
                    right=right_m.astype(np.uint8),
                    fg=fg,
                    bg=bg,
                )
                # QC overlay
                overlay = img.copy()
                draw = ImageDraw.Draw(overlay, "RGBA")
                if left_m.any():
                    draw.bitmap((0, 0), Image.fromarray((left_m * 255).astype(np.uint8)), fill=(255, 0, 0, 90))
                if right_m.any():
                    draw.bitmap((0, 0), Image.fromarray((right_m * 255).astype(np.uint8)), fill=(0, 0, 255, 90))
                overlay.save(out_dir / f"{sid}_qc.png")
                area_l = float(left_m.mean())
                area_r = float(right_m.mean())
                print(
                    f"[{cond}] {sid} ({i}/{len(items)}) L_area={area_l:.3f} R_area={area_r:.3f} "
                    f"L_score={left_s:.2f} R_score={right_s:.2f}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[{cond}] {sid} ERROR: {type(exc).__name__}: {exc}", flush=True)
    print("SAM masks done.", flush=True)


if __name__ == "__main__":
    main()
