from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(r"D:\Python\MMDIT\AAA_Experiment\experiment_4500")
OUT = ROOT / "binary_vqa_v2"
POOL_SEED = 8525
ORDER_SEEDS = {"GEMINI": 8526, "QWEN": 8527}


def main() -> None:
    native_map = json.loads(
        (ROOT / "base_vlm" / "key" / "blind_map.json").read_text(encoding="utf-8")
    )
    lsda_map = json.loads(
        (ROOT / "lsda_clean_vlm" / "key" / "blind_map.json").read_text(encoding="utf-8")
    )
    if len(native_map) != 900 or len(lsda_map) != 900:
        raise RuntimeError("expected 900 native and 900 LSDA records")

    source_records = []
    for condition, records, folder in (
        ("native_SS", native_map, ROOT / "base_vlm" / "blind"),
        ("lsda_clean", lsda_map, ROOT / "lsda_clean_vlm" / "blind"),
    ):
        for row in records:
            source_records.append(
                {**row, "condition": condition, "source_jpg": str(folder / f'{row["sample_id"]}.jpg')}
            )

    random.Random(POOL_SEED).shuffle(source_records)
    blind = OUT / "blind"
    key = OUT / "key"
    blind.mkdir(parents=True, exist_ok=True)
    key.mkdir(parents=True, exist_ok=True)
    records = []
    for index, row in enumerate(source_records, start=1):
        eval_id = f"eval_{index:04d}"
        source = Path(row.pop("source_jpg"))
        destination = blind / f"{eval_id}.jpg"
        image = Image.open(source).convert("RGB")
        if image.size != (1536, 512):
            image = image.resize((1536, 512), Image.Resampling.LANCZOS)
        image.save(destination, quality=85, optimize=True)
        records.append({"eval_id": eval_id, **row})

    (key / "blind_map.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ids = [row["eval_id"] for row in records]
    for rater, seed in ORDER_SEEDS.items():
        order = ids.copy()
        random.Random(seed).shuffle(order)
        (OUT / f"order_{rater}.json").write_text(
            json.dumps(order, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"images": len(records), "out": str(OUT)}))


if __name__ == "__main__":
    main()
