from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(r"D:\Python\MMDIT\AAA_Experiment\experiment_4500")
REQUIRED = ("standalone_A", "standalone_B", "candidate_A", "candidate_B", "shift", "structure")


def parse_payload(raw):
    if isinstance(raw, dict):
        payload = raw
    else:
        text = str(raw).strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            return None
        try:
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return payload if all(key in payload for key in REQUIRED) else None


def valid_ids(dataset: str, rater: str):
    valid = set()
    for path in sorted((ROOT / dataset / "ratings" / rater).glob("ratings_*.jsonl.raw")):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            try:
                envelope = json.loads(line)
            except json.JSONDecodeError:
                continue
            if parse_payload(envelope.get("raw")) is not None:
                valid.add(envelope["sample_id"])
    return valid


base_rows = list(csv.DictReader(
    (ROOT / "base_vlm" / "analysis" / "seed_level_results.csv").open(
        encoding="utf-8-sig", newline=""
    )
))
targets = [
    row["sample_id"] for row in base_rows
    if row.get("consensus_shift") == "True"
    and all(
        row.get(f"{model}_standalone_{entity}_correct") == "True"
        for model in ("qwen", "gemini") for entity in ("A", "B")
    )
]
assert len(targets) == 35, len(targets)

for dataset in ("original_vlm", "ra_vlm"):
    for rater in ("QWEN", "GEMINI"):
        valid = valid_ids(dataset, rater)
        missing = [sample_id for sample_id in targets if sample_id not in valid]
        path = ROOT / dataset / f"strict_target_retry_{rater}.json"
        path.write_text(json.dumps(missing, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"dataset": dataset, "rater": rater,
                          "targets": len(targets), "already_valid": len(targets) - len(missing),
                          "retry": len(missing)}))
