from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/science/wx/pry/AAA_Experiment/cultural100_v1")
CELL = 768


def fit(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((CELL, CELL - 42), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (CELL, CELL), (250, 248, 243))
    canvas.paste(image, ((CELL - image.width) // 2, 42 + (CELL - 42 - image.height) // 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("original", "ra"))
    args = parser.parse_args()
    method = args.method
    source_map = json.loads((ROOT / "base_vlm" / "key" / "blind_map.json").read_text(encoding="utf-8"))
    out = ROOT / ("original_vlm" if method == "original" else "ra_vlm")
    blind = out / "blind"; key_dir = out / "key"
    blind.mkdir(parents=True, exist_ok=True); key_dir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    records = []
    for row in source_map:
        native = row["native_task_id"]
        candidate_task = native.replace("native_SS", "lsda_original" if method == "original" else "lsda_ra")
        candidate_root = ROOT / ("lsda_original" if method == "original" else "lsda") / "images"
        paths = [
            ROOT / "base" / "images" / row["pair_id"] / "standalone_A" / f'{row["standalone_A_task_id"]}.png',
            ROOT / "base" / "images" / row["pair_id"] / "standalone_B" / f'{row["standalone_B_task_id"]}.png',
            candidate_root / row["pair_id"] / f"{candidate_task}.png",
        ]
        if not all(path.exists() for path in paths):
            raise FileNotFoundError([str(path) for path in paths if not path.exists()])
        sheet = Image.new("RGB", (CELL * 3, CELL), (250, 248, 243))
        draw = ImageDraw.Draw(sheet)
        for col, (path, label) in enumerate(zip(paths, ("A-alone", "B-alone", "Candidate AB"))):
            sheet.paste(fit(Image.open(path)), (col * CELL, 0))
            draw.text((col * CELL + 16, 10), label, fill=(25, 55, 82), font=font)
        sheet.save(blind / f'{row["sample_id"]}.jpg', quality=90, optimize=True)
        records.append({**row, "candidate_task_id": candidate_task, "method_key": method})
    (key_dir / "blind_map.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for rater, seed in (("QWEN", 521 if method == "original" else 621),
                        ("GEMINI", 522 if method == "original" else 622)):
        ids = [row["sample_id"] for row in records]
        random.Random(seed).shuffle(ids)
        (out / f"order_{rater}.json").write_text(json.dumps(ids, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"method": method, "samples": len(records), "out": str(out)}))


if __name__ == "__main__":
    main()
