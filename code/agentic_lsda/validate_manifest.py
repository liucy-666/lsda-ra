from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from schema import validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(payload)
    print(
        json.dumps(
            {
                "event": "manifest_valid",
                "task_count": len(payload["tasks"]),
                "samples": len({row["sample_id"] for row in payload["tasks"]}),
                "actions": Counter(row["action"]["kind"] for row in payload["tasks"]),
                "targets": Counter(row["action"]["target"] for row in payload["tasks"]),
            },
            ensure_ascii=False,
            default=dict,
        )
    )


if __name__ == "__main__":
    main()
