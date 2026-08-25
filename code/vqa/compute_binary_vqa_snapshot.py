from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(
    r"D:\Python\MMDIT\experiment\2026_8_25_EXP_1"
    r"\cultural100_records\experiment_4500\binary_vqa_v2"
)


def wilson(k: int, n: int) -> list[float | None]:
    if not n:
        return [None, None]
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def load_ratings(rater: str) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in sorted((ROOT / "ratings" / rater).glob("*.jsonl")):
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


def summarize(rater: str, blind_map: dict[str, dict]) -> dict:
    ratings = load_ratings(rater)
    by_condition: dict[str, list[dict]] = defaultdict(list)
    by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for eval_id, rating in ratings.items():
        key = blind_map.get(eval_id)
        if not key:
            continue
        condition = key["condition"]
        combined = {**key, **rating}
        by_condition[condition].append(combined)
        by_sample[key["sample_id"]][condition] = combined

    condition_summary = {}
    for condition in ("native_SS", "lsda_clean"):
        rows = by_condition.get(condition, [])
        n = len(rows)
        drift = sum(not bool(row["correct_binding"]) for row in rows)
        a_to_b = sum(bool(row["A_to_B_leakage"]) for row in rows)
        b_to_a = sum(bool(row["B_to_A_leakage"]) for row in rows)
        condition_summary[condition] = {
            "n": n,
            "correct": n - drift,
            "drift": drift,
            "drift_rate": drift / n if n else None,
            "drift_rate_wilson95": wilson(drift, n),
            "A_to_B_leakage": a_to_b,
            "B_to_A_leakage": b_to_a,
        }

    paired = [
        values
        for values in by_sample.values()
        if "native_SS" in values and "lsda_clean" in values
    ]
    restored = sum(
        (not pair["native_SS"]["correct_binding"])
        and pair["lsda_clean"]["correct_binding"]
        for pair in paired
    )
    worsened = sum(
        pair["native_SS"]["correct_binding"]
        and (not pair["lsda_clean"]["correct_binding"])
        for pair in paired
    )
    both_drift = sum(
        (not pair["native_SS"]["correct_binding"])
        and (not pair["lsda_clean"]["correct_binding"])
        for pair in paired
    )
    both_correct = sum(
        pair["native_SS"]["correct_binding"]
        and pair["lsda_clean"]["correct_binding"]
        for pair in paired
    )
    return {
        "valid_total": len(ratings),
        "conditions": condition_summary,
        "paired": {
            "n": len(paired),
            "restored_native_drift_to_lsda_correct": restored,
            "worsened_native_correct_to_lsda_drift": worsened,
            "both_drift": both_drift,
            "both_correct": both_correct,
        },
    }


def main() -> None:
    blind_rows = json.loads((ROOT / "key" / "blind_map.json").read_text(encoding="utf-8"))
    blind_map = {row["eval_id"]: row for row in blind_rows}
    result = {
        "snapshot_time": datetime.now().astimezone().isoformat(),
        "status": "interim_partial_blind_review",
        "unit": "one pair-seed image",
        "drift_definition": "left_choice != A or right_choice != B",
        "raters": {
            rater: summarize(rater, blind_map) for rater in ("QWEN", "GEMINI")
        },
    }
    output = ROOT / "interim_rates_snapshot.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
