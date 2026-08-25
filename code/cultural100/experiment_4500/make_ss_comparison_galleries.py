from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"D:\Python\MMDIT\AAA_Experiment\experiment_4500")
sys.path.insert(0, str(ROOT))
import analyze_base_vlm as analyzer

OUT = ROOT / "inspection"
PAGE_W = 2400
MARGIN = 28
LABEL_H = 52
GAP = 24
ROWS = 3


def font(size):
    path = Path(r"C:\Windows\Fonts\arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def strict_targets(base):
    return [
        sample_id for sample_id, row in base.items()
        if row.get("consensus_shift") == "True"
        and all(
            row.get(f"{model}_standalone_{entity}_correct") == "True"
            for model in ("qwen", "gemini") for entity in ("A", "B")
        )
    ]


def image_success(payload):
    structure = payload["structure"]
    return (
        bool(payload["candidate_A"]["correct"])
        and bool(payload["candidate_B"]["correct"])
        and bool(structure["two_objects_present"])
        and bool(structure["left_right_correct"])
        and not bool(structure["merged_or_touching"])
        and int(structure["extra_objects"]) == 0
    )


def confirmed_failures(dataset, targets):
    analyzer.ROOT = ROOT / dataset
    ratings = {rater: analyzer.load_rater(rater)[0] for rater in ("QWEN", "GEMINI")}
    return [
        sample_id for sample_id in targets
        if not image_success(ratings["QWEN"][sample_id])
        and not image_success(ratings["GEMINI"][sample_id])
    ]


def four_panel(dataset, sample_id):
    base = Image.open(ROOT / "base_vlm" / "blind" / f"{sample_id}.jpg").convert("RGB")
    method = Image.open(ROOT / dataset / "blind" / f"{sample_id}.jpg").convert("RGB")
    cell = base.width // 3
    panels = [
        base.crop((0, 0, cell, base.height)),
        base.crop((cell, 0, 2 * cell, base.height)),
        base.crop((2 * cell, 0, 3 * cell, base.height)),
        method.crop((2 * cell, 0, 3 * cell, method.height)),
    ]
    labels = ("A-alone", "B-alone", "Native SS", "LSDA" if dataset == "original_vlm" else "LSDA-RA")
    sheet = Image.new("RGB", (cell * 4, base.height), "#FAF6EE")
    draw = ImageDraw.Draw(sheet)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        sheet.paste(panel, (index * cell, 0))
        draw.rectangle((index * cell, 0, index * cell + 220, 40), fill="#FAF6EE")
        draw.text((index * cell + 12, 7), label, fill="#1F4E79", font=font(24))
    return sheet


def make_pages(dataset, ids, base_rows):
    target = OUT / f"strict_failed_ss_{dataset}"
    target.mkdir(parents=True, exist_ok=True)
    tile_w = PAGE_W - 2 * MARGIN
    tile_h = round(tile_w / 4)
    page_h = 2 * MARGIN + ROWS * (LABEL_H + tile_h) + (ROWS - 1) * GAP
    index_rows = []
    for page_index in range(math.ceil(len(ids) / ROWS)):
        batch = ids[page_index * ROWS : (page_index + 1) * ROWS]
        page = Image.new("RGB", (PAGE_W, page_h), "#FAF6EE")
        draw = ImageDraw.Draw(page)
        for row_index, sample_id in enumerate(batch):
            y = MARGIN + row_index * (LABEL_H + tile_h + GAP)
            meta = base_rows[sample_id]
            title = f"{sample_id} | {meta.get('pair_id')} | seed {meta.get('latent_seed')}"
            draw.text((MARGIN, y + 7), title, fill="#1F4E79", font=font(30))
            sheet = four_panel(dataset, sample_id).resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            page.paste(sheet, (MARGIN, y + LABEL_H))
            index_rows.append({
                "method": dataset,
                "sample_id": sample_id,
                "pair_id": meta.get("pair_id"),
                "seed": meta.get("latent_seed"),
                "page": page_index + 1,
            })
        page.save(target / f"page_{page_index + 1:02d}.jpg", quality=94, subsampling=0)
    with (target / "index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "sample_id", "pair_id", "seed", "page"])
        writer.writeheader()
        writer.writerows(index_rows)
    print(dataset, len(ids), math.ceil(len(ids) / ROWS), target)


def main():
    base = read_rows(ROOT / "base_vlm" / "analysis" / "seed_level_results.csv")
    targets = strict_targets(base)
    for dataset in ("original_vlm", "ra_vlm"):
        make_pages(dataset, confirmed_failures(dataset, targets), base)


if __name__ == "__main__":
    main()
