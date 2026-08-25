from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor


ROOT = Path("/science/wx/pry/AAA_Experiment/cultural100_v1")
MANIFEST = ROOT / "lsda_manifest.json"
SAM = Path("/science/wx/pry/baseline_v2/models/sam-vit-base")
SLOTS = {"A": (30, 70, 500, 960), "B": (524, 70, 994, 960)}


def source_path(row: dict, entity: str) -> Path:
    image_type = f"standalone_{entity}"
    task_id = row[f"standalone_{entity}_task_id"]
    return ROOT / "base" / "images" / row["pair_id"] / image_type / f"{task_id}.png"


def segment(processor, model, image: Image.Image) -> tuple[np.ndarray, float, int]:
    w, h = image.size
    box = (8, 8, w - 8, h - 8)
    inputs = processor(image, input_boxes=[[list(box)]], return_tensors="pt")
    model_inputs = {
        key: value.to("cuda")
        for key, value in inputs.items()
        if key not in ("original_sizes", "reshaped_input_sizes")
    }
    with torch.inference_mode():
        outputs = model(**model_inputs)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )[0]
    scores = outputs.iou_scores[0, 0].detach().cpu()
    candidates = []
    for i in range(masks.shape[1]):
        mask = masks[0, i].numpy().astype(bool)
        area = float(mask.mean())
        candidates.append((i, mask, area, float(scores[i].item())))
    plausible = [x for x in candidates if 0.01 <= x[2] <= 0.88]
    chosen = max(plausible or candidates, key=lambda x: x[3])
    return chosen[1], chosen[3], chosen[0]


def fit_to_slot(mask: np.ndarray, slot: tuple[int, int, int, int]) -> np.ndarray:
    ys, xs = np.where(mask)
    if not len(xs):
        raise RuntimeError("empty standalone mask")
    crop = Image.fromarray(
        (mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] * 255).astype(np.uint8), mode="L"
    )
    x0, y0, x1, y1 = slot
    scale = min((x1 - x0) / crop.width, (y1 - y0) / crop.height)
    new_w = max(1, round(crop.width * scale))
    new_h = max(1, round(crop.height * scale))
    resized = crop.resize((new_w, new_h), Image.Resampling.NEAREST)
    canvas = Image.new("L", (1024, 1024), 0)
    px = x0 + ((x1 - x0) - new_w) // 2
    py = y0 + ((y1 - y0) - new_h) // 2
    canvas.paste(resized, (px, py))
    return np.asarray(canvas, dtype=np.uint8) > 127


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=int)
    parser.add_argument("nshards", type=int)
    args = parser.parse_args()
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tasks = [row for i, row in enumerate(payload["tasks"]) if i % args.nshards == args.shard]
    processor = SamProcessor.from_pretrained(SAM, local_files_only=True)
    model = SamModel.from_pretrained(SAM, local_files_only=True).to("cuda").eval()

    for row in tasks:
        out = ROOT / "lsda" / "masks" / row["task_id"]
        summary = out / "mask_summary.json"
        if summary.exists():
            print(json.dumps({"event": "skip", "task_id": row["task_id"]}), flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        records = []
        for index, entity in enumerate(("A", "B"), start=1):
            src = source_path(row, entity)
            image = Image.open(src).convert("RGB")
            raw, score, candidate = segment(processor, model, image)
            fitted = fit_to_slot(raw, SLOTS[entity])
            Image.fromarray((fitted * 255).astype(np.uint8), mode="L").save(
                out / f"entity_{index}.png"
            )
            records.append({
                "entity": entity,
                "source": str(src),
                "source_prompt_type": "standalone_short",
                "sam_candidate": candidate,
                "sam_iou": score,
                "raw_area": float(raw.mean()),
                "fitted_area": float(fitted.mean()),
                "slot": SLOTS[entity],
            })
        summary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"event": "done", "task_id": row["task_id"]}), flush=True)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
