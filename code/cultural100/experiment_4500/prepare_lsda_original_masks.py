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
BOXES = {"A": (8, 8, 520, 1016), "B": (504, 8, 1016, 1016)}


def native_path(row: dict) -> Path:
    return (
        ROOT / "base" / "images" / row["pair_id"] / "native_SS"
        / f'{row["native_task_id"]}.png'
    )


def segment(processor, model, image: Image.Image, box) -> tuple[np.ndarray, float, int]:
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
    plausible = [x for x in candidates if 0.005 <= x[2] <= 0.60]
    chosen = max(plausible or candidates, key=lambda x: x[3])
    return chosen[1], chosen[3], chosen[0]


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
        out = ROOT / "lsda_original" / "masks" / row["task_id"]
        summary = out / "mask_summary.json"
        if summary.exists():
            print(json.dumps({"event": "skip", "task_id": row["task_id"]}), flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        source = native_path(row)
        image = Image.open(source).convert("RGB")
        records = []
        masks = []
        for index, entity in enumerate(("A", "B"), start=1):
            mask, score, candidate = segment(processor, model, image, BOXES[entity])
            masks.append(mask)
            Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
                out / f"entity_{index}.png"
            )
            records.append({
                "entity": entity,
                "source": str(source),
                "source_type": "native_SS_self_segmentation",
                "sam_box": BOXES[entity],
                "sam_candidate": candidate,
                "sam_iou": score,
                "area": float(mask.mean()),
            })
        records.append({"raw_overlap_fraction": float((masks[0] & masks[1]).mean())})
        summary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"event": "done", "task_id": row["task_id"]}), flush=True)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
