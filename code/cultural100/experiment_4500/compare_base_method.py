from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def as_bool(value):
    if value == "True":
        return True
    if value == "False":
        return False
    return None


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rating_state(row):
    qwen = as_bool(row.get("qwen_shift"))
    gemini = as_bool(row.get("gemini_shift"))
    if qwen is None or gemini is None:
        return None
    if qwen and gemini:
        return "fail"
    if not qwen and not gemini:
        return "success"
    return "disagree"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base = read_rows(args.base / "analysis" / "seed_level_results.csv")
    method = read_rows(args.method / "analysis" / "seed_level_results.csv")
    rows = []
    for sample_id in sorted(set(base) & set(method)):
        b, m = base[sample_id], method[sample_id]
        bs = as_bool(b.get("consensus_shift"))
        ms = as_bool(m.get("consensus_shift"))
        base_state = rating_state(b)
        method_state = rating_state(m)
        row = {
            "sample_id": sample_id,
            "pair_id": b.get("pair_id"),
            "seed": b.get("seed"),
            "base_consensus_shift": bs,
            "method_consensus_shift": ms,
            "paired_valid": base_state is not None and method_state is not None,
            "base_state": base_state,
            "method_state": method_state,
            "strict_rescue": base_state == "fail" and method_state == "success",
            "persistent_failure": base_state == "fail" and method_state == "fail",
            "strict_harm": base_state == "success" and method_state == "fail",
            "stable_success": base_state == "success" and method_state == "success",
            "fail_to_disagree": base_state == "fail" and method_state == "disagree",
            "disagree_to_fail": base_state == "disagree" and method_state == "fail",
        }
        for entity in ("A", "B"):
            bq = as_float(b.get(f"qwen_candidate_{entity}_score"))
            bg = as_float(b.get(f"gemini_candidate_{entity}_score"))
            mq = as_float(m.get(f"qwen_candidate_{entity}_score"))
            mg = as_float(m.get(f"gemini_candidate_{entity}_score"))
            row[f"base_{entity}_identity_mean"] = ((bq + bg) / 2) if bq is not None and bg is not None else None
            row[f"method_{entity}_identity_mean"] = ((mq + mg) / 2) if mq is not None and mg is not None else None
            row[f"delta_{entity}_identity"] = (
                row[f"method_{entity}_identity_mean"] - row[f"base_{entity}_identity_mean"]
                if row[f"base_{entity}_identity_mean"] is not None and row[f"method_{entity}_identity_mean"] is not None
                else None
            )
        rows.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (args.out / "paired_seed_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

    valid = [row for row in rows if row["paired_valid"]]
    base_fail = [row for row in valid if row["base_state"] == "fail"]
    summary = {
        "paired_valid": len(valid),
        "base_shift_count": sum(row["base_consensus_shift"] for row in valid),
        "method_shift_count": sum(row["method_consensus_shift"] for row in valid),
        "strict_rescue_count": sum(row["strict_rescue"] for row in valid),
        "strict_rescue_rate_among_base_failures": (
            sum(row["strict_rescue"] for row in valid) / len(base_fail) if base_fail else None
        ),
        "persistent_failure_count": sum(row["persistent_failure"] for row in valid),
        "strict_harm_count": sum(row["strict_harm"] for row in valid),
        "stable_success_count": sum(row["stable_success"] for row in valid),
        "fail_to_disagree_count": sum(row["fail_to_disagree"] for row in valid),
        "disagree_to_fail_count": sum(row["disagree_to_fail"] for row in valid),
        "rule": "Strict rescue requires dual-VLM failure for Native and dual-VLM success for the method on the identical Pair x seed sample_id; rater disagreement is kept separate.",
    }
    (args.out / "paired_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
