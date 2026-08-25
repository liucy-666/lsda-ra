from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for rater in ("QWEN", "GEMINI"):
        valid = set()
        for path in sorted((args.root / "ratings" / rater).glob("ratings_*.jsonl.raw")):
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                try:
                    envelope = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if parse_payload(envelope.get("raw")) is not None:
                    valid.add(envelope.get("sample_id"))
        order = json.loads((args.root / f"order_{rater}.json").read_text(encoding="utf-8"))
        missing = [sample_id for sample_id in order if sample_id not in valid]
        out = args.root / f"retry_order_{rater}.json"
        out.write_text(json.dumps(missing, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"root": str(args.root), "rater": rater, "valid": len(valid), "missing": len(missing)}))


if __name__ == "__main__":
    main()
