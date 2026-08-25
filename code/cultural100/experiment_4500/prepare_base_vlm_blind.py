from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REMOTE_ROOT = Path("/science/wx/pry/AAA_Experiment/cultural100_v1")
LOCAL_ROOT = Path(r"D:\Python\MMDIT\AAA_Experiment\experiment_4500\base_vlm")
CELL = 768
SHUFFLE_SEED = 420


def fit(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((CELL, CELL - 42), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (CELL, CELL), (250, 248, 243))
    x = (CELL - image.width) // 2
    y = 42 + (CELL - 42 - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REMOTE_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root
    out = args.out or (root / "base_vlm")
    manifest = json.loads((root / "base_manifest.json").read_text(encoding="utf-8"))
    lsda_manifest = json.loads((root / "lsda_manifest.json").read_text(encoding="utf-8"))
    task_index = {row["task_id"]: row for row in manifest["tasks"]}
    pair_source = json.loads(
        (Path(r"D:\Python\MMDIT\AAA_Experiment\cultural_pairs_100.json")
         if str(root).startswith("D:") else root / "cultural_pairs_100.json").read_text(encoding="utf-8")
    )
    samples = list(lsda_manifest["tasks"])
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(samples)
    blind = out / "blind"
    key = out / "key"
    blind.mkdir(parents=True, exist_ok=True)
    key.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    labels = ("A-alone", "B-alone", "Native SS")
    for number, row in enumerate(samples, start=1):
        sample_id = f"sample_{number:04d}"
        paths = [
            root / "base" / "images" / row["pair_id"] / "standalone_A"
            / f'{row["standalone_A_task_id"]}.png',
            root / "base" / "images" / row["pair_id"] / "standalone_B"
            / f'{row["standalone_B_task_id"]}.png',
            root / "base" / "images" / row["pair_id"] / "native_SS"
            / f'{row["native_task_id"]}.png',
        ]
        sheet = Image.new("RGB", (CELL * 3, CELL), (250, 248, 243))
        draw = ImageDraw.Draw(sheet)
        for col, (path, label) in enumerate(zip(paths, labels)):
            sheet.paste(fit(Image.open(path)), (col * CELL, 0))
            draw.text((col * CELL + 16, 10), label, fill=(25, 55, 82), font=font)
        sheet.save(blind / f"{sample_id}.jpg", quality=90, optimize=True)
        pair = pair_source[row["pair_index"] - 1]
        records.append({
            "sample_id": sample_id,
            "pair_id": row["pair_id"],
            "pair_index": row["pair_index"],
            "seed_group": row["seed_group"],
            "replicate": row["replicate"],
            "latent_seed": row["latent_seed"],
            "entity_A": row["entity_A_prompt"],
            "entity_B": row["entity_B_prompt"],
            "entity_A_diagnostic": pair["文化物体A的长文本描述"],
            "entity_B_diagnostic": pair["文化物体B的长文本描述"],
            "standalone_A_task_id": row["standalone_A_task_id"],
            "standalone_B_task_id": row["standalone_B_task_id"],
            "native_task_id": row["native_task_id"],
        })
    (key / "blind_map.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for rater_id, seed in (("QWEN", 421), ("GEMINI", 422)):
        ids = [row["sample_id"] for row in records]
        random.Random(seed).shuffle(ids)
        (out / f"order_{rater_id}.json").write_text(
            json.dumps(ids, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"samples": len(records), "out": str(out)}))


if __name__ == "__main__":
    main()
