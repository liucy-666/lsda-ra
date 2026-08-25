from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(r"D:\Python\MMDIT\AAA_Experiment\experiment_4500")
DATASETS = ("base_vlm", "original_vlm", "ra_vlm")
RATERS = ("QWEN", "GEMINI")
REQUIRED = ("standalone_A", "standalone_B", "candidate_A", "candidate_B", "shift", "structure")
LOG = ROOT / "vlm_progress_10min.jsonl"


def valid_payload(raw):
    if isinstance(raw, dict):
        payload = raw
    else:
        text = str(raw).strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            return False
        try:
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return False
    return all(key in payload for key in REQUIRED)


def snapshot():
    result = {"time": datetime.now().astimezone().isoformat(timespec="seconds"), "datasets": {}}
    all_done = True
    for dataset in DATASETS:
        root = ROOT / dataset
        rec = {"blind_images": len(list((root / "blind").glob("*.jpg"))) if root.exists() else 0}
        for rater in RATERS:
            valid, missing, malformed, seen = set(), 0, 0, set()
            for path in sorted((root / "ratings" / rater).glob("ratings_*.jsonl.raw")):
                for line in path.read_text(encoding="utf-8-sig").splitlines():
                    try:
                        envelope = json.loads(line)
                    except json.JSONDecodeError:
                        malformed += 1
                        continue
                    sample_id = envelope.get("sample_id")
                    seen.add(sample_id)
                    raw = envelope.get("raw")
                    if raw == "MISSING":
                        missing += 1
                    elif valid_payload(raw):
                        valid.add(sample_id)
                    else:
                        malformed += 1
            rec[rater] = {"valid_unique": len(valid), "seen_unique": len(seen),
                          "missing_rows": missing, "malformed_rows": malformed}
            all_done = all_done and len(valid) == 900
        result["datasets"][dataset] = rec
        all_done = all_done and rec["blind_images"] == 900
    return result, all_done


def main():
    while True:
        row, done = snapshot()
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if done:
            break
        time.sleep(600)


if __name__ == "__main__":
    main()
