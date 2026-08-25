from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(
    r"D:\Python\MMDIT\experiment\2026_8_25_EXP_1"
    r"\cultural100_records\experiment_4500\binary_vqa_v2"
)
BLIND = Path(r"D:\Python\MMDIT\data\Blind\2026_8_25_EXP_1")
IMAGE_OUT = Path(r"D:\Python\MMDIT\data\Inspection\2026_8_25_EXP_1")
TABLE_OUT = ROOT / "inspection"


def load_ratings() -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in sorted((ROOT / "ratings" / "QWEN").glob("*.jsonl")):
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                eval_id = row.get("eval_id")
                if eval_id and eval_id not in records:
                    records[eval_id] = row
    return records


def fit(image: Image.Image, width: int) -> Image.Image:
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def main() -> None:
    key_rows = json.loads((ROOT / "key" / "blind_map.json").read_text(encoding="utf-8"))
    key = {row["eval_id"]: row for row in key_rows}
    ratings = load_ratings()
    by_sample: dict[str, dict[str, tuple[dict, dict]]] = defaultdict(dict)
    for eval_id, rating in ratings.items():
        meta = key.get(eval_id)
        if meta:
            by_sample[meta["sample_id"]][meta["condition"]] = (meta, rating)

    cases = []
    for sample_id, conditions in by_sample.items():
        if not {"native_SS", "lsda_clean"}.issubset(conditions):
            continue
        native_meta, native = conditions["native_SS"]
        lsda_meta, lsda = conditions["lsda_clean"]
        if native["correct_binding"] and not lsda["correct_binding"]:
            cases.append(
                {
                    "sample_id": sample_id,
                    "pair_id": native_meta["pair_id"],
                    "latent_seed": native_meta["latent_seed"],
                    "entity_A": native_meta["entity_A"],
                    "entity_B": native_meta["entity_B"],
                    "native_eval_id": native_meta["eval_id"],
                    "lsda_eval_id": lsda_meta["eval_id"],
                    "native_left_choice": native["left_choice"],
                    "native_right_choice": native["right_choice"],
                    "lsda_left_choice": lsda["left_choice"],
                    "lsda_right_choice": lsda["right_choice"],
                    "lsda_left_reason": lsda.get("left_reason", ""),
                    "lsda_right_reason": lsda.get("right_reason", ""),
                }
            )
    cases.sort(key=lambda row: (row["pair_id"], row["latent_seed"]))

    TABLE_OUT.mkdir(parents=True, exist_ok=True)
    with (TABLE_OUT / "qwen_ss_correct_lsda_drift.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)

    IMAGE_OUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    panel_width = 760
    row_gap = 18
    header_height = 44
    rows_per_sheet = 5
    for sheet_index in range(0, len(cases), rows_per_sheet):
        chunk = cases[sheet_index : sheet_index + rows_per_sheet]
        rendered = []
        for case in chunk:
            native = fit(Image.open(BLIND / f"{case['native_eval_id']}.jpg").convert("RGB"), panel_width)
            lsda = fit(Image.open(BLIND / f"{case['lsda_eval_id']}.jpg").convert("RGB"), panel_width)
            rendered.append((case, native, lsda))
        row_height = max(max(a.height, b.height) for _, a, b in rendered) + header_height
        canvas = Image.new(
            "RGB",
            (panel_width * 2 + 32, row_height * len(rendered) + row_gap * (len(rendered) - 1)),
            (250, 246, 238),
        )
        draw = ImageDraw.Draw(canvas)
        y = 0
        for case, native, lsda in rendered:
            direction = []
            if case["lsda_left_choice"] == "B":
                direction.append("left->B")
            if case["lsda_right_choice"] == "A":
                direction.append("right->A")
            title = (
                f"{case['pair_id']} | seed {case['latent_seed']} | "
                f"LSDA drift: {', '.join(direction)}"
            )
            draw.text((8, y + 5), title, fill=(31, 78, 121), font=font)
            draw.text((8, y + 23), f"Native SS ({case['native_eval_id']})", fill=(40, 40, 40), font=font)
            draw.text((panel_width + 24, y + 23), f"LSDA ({case['lsda_eval_id']})", fill=(200, 85, 61), font=font)
            canvas.paste(native, (8, y + header_height))
            canvas.paste(lsda, (panel_width + 24, y + header_height))
            y += row_height + row_gap
        output = IMAGE_OUT / f"qwen_ss_correct_lsda_drift_{sheet_index // rows_per_sheet + 1}.jpg"
        canvas.save(output, quality=94, subsampling=0)
        print(output)
    print(f"cases={len(cases)}")


if __name__ == "__main__":
    main()
