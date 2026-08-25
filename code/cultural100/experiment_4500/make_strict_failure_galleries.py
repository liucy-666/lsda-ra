from __future__ import annotations

import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"D:\Python\MMDIT\AAA_Experiment\experiment_4500")
OUT = ROOT / "inspection"
PAGE_W = 2400
MARGIN = 28
GAP = 24
COLS = 2
ROWS = 3
LABEL_H = 54
TILE_W = (PAGE_W - 2 * MARGIN - GAP) // COLS


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def strict_targets(base):
    return [
        sample_id
        for sample_id, row in base.items()
        if row.get("consensus_shift") == "True"
        and all(
            row.get(f"{model}_standalone_{entity}_correct") == "True"
            for model in ("qwen", "gemini")
            for entity in ("A", "B")
        )
    ]


def failed_ids(base, method):
    return [
        sample_id
        for sample_id in strict_targets(base)
        if method[sample_id].get("qwen_shift") == "True"
        and method[sample_id].get("gemini_shift") == "True"
    ]


def font(size):
    path = Path(r"C:\Windows\Fonts\arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def make_pages(dataset: str, ids: list[str], base):
    source = ROOT / dataset / "blind"
    target = OUT / f"strict_failed_{dataset}"
    target.mkdir(parents=True, exist_ok=True)
    meta = []
    page_size = COLS * ROWS
    for page_index in range(math.ceil(len(ids) / page_size)):
        batch = ids[page_index * page_size : (page_index + 1) * page_size]
        rendered = []
        for sample_id in batch:
            image = Image.open(source / f"{sample_id}.jpg").convert("RGB")
            ratio = TILE_W / image.width
            image = image.resize((TILE_W, round(image.height * ratio)), Image.Resampling.LANCZOS)
            rendered.append((sample_id, image))
        tile_h = max(image.height for _, image in rendered)
        page_h = 2 * MARGIN + ROWS * (LABEL_H + tile_h) + (ROWS - 1) * GAP
        page = Image.new("RGB", (PAGE_W, page_h), "#FAF6EE")
        draw = ImageDraw.Draw(page)
        for slot, (sample_id, image) in enumerate(rendered):
            row_index, col_index = divmod(slot, COLS)
            x = MARGIN + col_index * (TILE_W + GAP)
            y = MARGIN + row_index * (LABEL_H + tile_h + GAP)
            record = base[sample_id]
            label = f"{sample_id} | {record.get('pair_id')} | seed {record.get('latent_seed')}"
            draw.text((x, y + 8), label, fill="#1F4E79", font=font(30))
            page.paste(image, (x, y + LABEL_H))
            meta.append({
                "method": dataset,
                "sample_id": sample_id,
                "pair_id": record.get("pair_id"),
                "seed": record.get("latent_seed"),
                "page": page_index + 1,
            })
        page.save(target / f"page_{page_index + 1:02d}.jpg", quality=94, subsampling=0)
    with (target / "index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "sample_id", "pair_id", "seed", "page"])
        writer.writeheader()
        writer.writerows(meta)
    print(dataset, len(ids), math.ceil(len(ids) / page_size), target)


def main():
    base = read_rows(ROOT / "base_vlm" / "analysis" / "seed_level_results.csv")
    for dataset in ("original_vlm", "ra_vlm"):
        method = read_rows(ROOT / dataset / "analysis" / "seed_level_results.csv")
        make_pages(dataset, failed_ids(base, method), base)


if __name__ == "__main__":
    main()
