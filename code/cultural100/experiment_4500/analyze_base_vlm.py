from __future__ import annotations

import csv
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_ROOT = Path(r"D:\Python\MMDIT\AAA_Experiment\experiment_4500\base_vlm")
ROOT = DEFAULT_ROOT
RATERS = ("QWEN", "GEMINI")


def parse_json(text):
    if isinstance(text, dict):
        return text
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object")
    return json.loads(text[start:end + 1])


def load_rater(rater):
    records, errors = {}, []
    for path in sorted((ROOT / "ratings" / rater).glob("ratings_*.jsonl.raw")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                envelope = json.loads(line)
                sample_id = envelope["sample_id"]
                if sample_id in records:
                    continue
                payload = parse_json(envelope["raw"])
                required = ("standalone_A", "standalone_B", "candidate_A", "candidate_B",
                            "shift", "structure")
                if any(key not in payload for key in required):
                    raise ValueError("missing required key")
                records[sample_id] = payload
            except Exception as exc:
                errors.append({"file": str(path), "line": line_no, "error": str(exc)})
    return records, errors


def affected(payload, entity):
    shift = payload["shift"]
    if entity == "A":
        return bool(shift.get("A_loss")) or bool(shift.get("B_to_A_leak"))
    return bool(shift.get("B_loss")) or bool(shift.get("A_to_B_leak"))


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    ROOT = args.root
    key_rows = json.loads((ROOT / "key" / "blind_map.json").read_text(encoding="utf-8"))
    key = {row["sample_id"]: row for row in key_rows}
    ratings, all_errors = {}, []
    for rater in RATERS:
        ratings[rater], errors = load_rater(rater)
        all_errors.extend({"rater": rater, **row} for row in errors)

    rows = []
    for sample_id, meta in key.items():
        q = ratings["QWEN"].get(sample_id)
        g = ratings["GEMINI"].get(sample_id)
        row = {**meta, "qwen_valid": q is not None, "gemini_valid": g is not None}
        if q is None or g is None:
            row.update({"consensus_shift": None, "attribution": "missing_rating"})
            rows.append(row)
            continue
        consensus_shift = bool(q["shift"].get("any_shift")) and bool(g["shift"].get("any_shift"))
        row["qwen_shift"] = bool(q["shift"].get("any_shift"))
        row["gemini_shift"] = bool(g["shift"].get("any_shift"))
        row["consensus_shift"] = consensus_shift
        for entity in ("A", "B"):
            row[f"qwen_standalone_{entity}_correct"] = bool(q[f"standalone_{entity}"]["correct"])
            row[f"gemini_standalone_{entity}_correct"] = bool(g[f"standalone_{entity}"]["correct"])
            row[f"qwen_candidate_{entity}_score"] = int(q[f"candidate_{entity}"]["identity_score"])
            row[f"gemini_candidate_{entity}_score"] = int(g[f"candidate_{entity}"]["identity_score"])
            row[f"consensus_{entity}_affected"] = affected(q, entity) and affected(g, entity)

        if not consensus_shift:
            row["attribution"] = "no_consensus_shift"
        else:
            entities = [e for e in ("A", "B") if row[f"consensus_{e}_affected"]]
            if not entities:
                row["attribution"] = "shift_location_disagreement"
            else:
                states = []
                for entity in entities:
                    a = row[f"qwen_standalone_{entity}_correct"]
                    b = row[f"gemini_standalone_{entity}_correct"]
                    states.append("correct" if a and b else "wrong" if not a and not b else "disagree")
                if all(state == "correct" for state in states):
                    row["attribution"] = "composition_mechanism_failure"
                elif "disagree" in states or ("correct" in states and "wrong" in states):
                    row["attribution"] = "mixed_or_uncertain"
                else:
                    row["attribution"] = "knowledge_insufficiency"
        rows.append(row)

    out = ROOT / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (out / "seed_level_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    shifted = [row for row in rows if row.get("consensus_shift") is True]
    with (out / "shifted_seed_list.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(shifted)
    counts = defaultdict(int)
    for row in shifted:
        counts[row["attribution"]] += 1
    valid = sum(row.get("consensus_shift") is not None for row in rows)
    summary = {
        "total_seed_samples": len(rows),
        "dual_vlm_valid_samples": valid,
        "consensus_attribute_shift_count": len(shifted),
        "consensus_attribute_shift_frequency": len(shifted) / valid if valid else None,
        "attribution_counts": dict(counts),
        "rule": "A seed counts as shifted only when Qwen-VL and Gemini both set any_shift=true",
        "errors": all_errors,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
