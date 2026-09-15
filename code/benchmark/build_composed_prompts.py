"""Create same-category, cross-cultural two-object prompts from KB records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union


EXCLUDED = {
    "landmark", "landmarks", "landscape", "house", "architecture",
    "celebration", "sport", "sports", "event", "place",
}
TEMPLATE = (
    "Neutral studio background: {a} on the left, {b} on the right; both fully "
    "visible, separate, and similar in size."
)


def _slug(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _identity(row: Mapping[str, Any]) -> str:
    return str(row.get("key") or row.get("concept") or "").lower()


def _pair_key(a: Mapping[str, Any], b: Mapping[str, Any]) -> str:
    left, right = sorted((_identity(a), _identity(b)))
    return left + "|" + right


def _prompt_entity(row: Mapping[str, Any]) -> str:
    concept = _slug(row.get("concept"))
    culture = _slug(row.get("culture"))
    knowledge = _slug(row.get("knowledge_text"))
    phrase = f"{concept} from {culture}" if culture else concept
    if knowledge:
        phrase += f" ({knowledge})"
    return phrase


def compose_pairs(
    records: Iterable[Mapping[str, Any]], *, max_pairs: int = 3000, seed: int = 0
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        category = _slug(row.get("category")).lower()
        if not row.get("concept") or not row.get("culture") or category in EXCLUDED:
            continue
        if row.get("status") not in (None, "ok", "warning"):
            continue
        groups[category].append(row)

    candidates: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for category, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: (_identity(row), _slug(row.get("culture"))))
        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                if _slug(left.get("culture")).lower() == _slug(right.get("culture")).lower():
                    continue
                key = f"{category}|{_pair_key(left, right)}"
                candidates[key] = (left, right)

    def rank(key: str) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()
        return digest, key

    selected = sorted(candidates, key=rank)[: max(0, int(max_pairs))]
    output = []
    for index, key in enumerate(selected):
        left, right = candidates[key]
        if (_slug(left.get("culture")).lower(), _identity(left)) > (
            _slug(right.get("culture")).lower(), _identity(right)
        ):
            left, right = right, left
        pair_id = f"kb-{index:05d}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"
        output.append(
            {
                "pair_id": pair_id,
                "category": _slug(left.get("category")),
                "a": dict(left),
                "b": dict(right),
                "prompt": TEMPLATE.format(a=_prompt_entity(left), b=_prompt_entity(right)),
                "controls": {
                    "same_category": True,
                    "cross_culture": True,
                    "position": "a_left_b_right",
                },
            }
        )
    return output


def read_jsonl(path: Union[str, Path]) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rows = compose_pairs(read_jsonl(args.kb), max_pairs=args.max_pairs, seed=args.seed)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"pairs": len(rows), "out": str(destination)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
