"""Normalize CUBE/TU concept rows into one stable KB input schema."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union


CATEGORY_CROSSWALK = {
    "cuisine": "food",
    "food": "food",
    "art": "visual_arts",
    "visual arts": "visual_arts",
    "visual_arts": "visual_arts",
    "clothing": "clothing",
    "dress": "clothing",
    "utensil": "utensil",
    "music": "music",
    "musical instrument": "music",
}


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def canonical_category(value: Any) -> str:
    raw = _text(value).lower().replace("-", " ").replace("/", " ")
    return CATEGORY_CROSSWALK.get(raw, re.sub(r"\s+", "_", raw) or "unknown")


def _first(row: Mapping[str, Any], names: Iterable[str]) -> str:
    lower = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        value = lower.get(name.lower())
        if _text(value):
            return _text(value)
    return ""


def stable_key(concept: str, culture: str, category: str, source: str) -> str:
    raw = "|".join((concept, culture, category, source)).lower()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def normalize_record(
    row: Mapping[str, Any], *, source: str = "unknown", llm: Optional[Any] = None
) -> dict[str, Any]:
    """Normalize one source row without inventing missing artifact facts.

    ``llm`` is an optional object exposing ``complete_json(system, user)``. It
    may fill missing semantic fields, but failures fall back to deterministic
    extraction and are never allowed to stop a batch.
    """
    concept = _first(row, ("concept", "entity", "name", "label", "artifact"))
    culture = _first(row, ("culture", "tradition", "country", "nation", "origin"))
    category = canonical_category(_first(row, ("category", "domain", "type", "class")))
    object_type = _first(row, ("object_type", "object", "head", "noun"))
    if llm is not None and (not concept or not culture or category == "unknown" or not object_type):
        try:
            result = llm.complete_json(
                "Return JSON with concept, culture, object_type, and category. "
                "Use only the supplied row; do not invent an artifact.",
                json.dumps(dict(row), ensure_ascii=False),
            )
            if isinstance(result, Mapping):
                concept = concept or _text(result.get("concept"))
                culture = culture or _text(result.get("culture"))
                if category == "unknown":
                    category = canonical_category(result.get("category"))
                object_type = object_type or _text(result.get("object_type"))
        except Exception:
            pass
    object_type = object_type or concept
    source = _text(source) or "unknown"
    out: dict[str, Any] = {
        "key": _first(row, ("key",)) or stable_key(concept, culture, category, source),
        "concept": concept,
        "culture": culture,
        "category": category,
        "object_type": object_type,
        "source": source,
    }
    for name in ("source_id", "qid", "url"):
        value = _first(row, (name,))
        if value:
            out[name] = value
    return out


def read_records(path: Union[str, Path], *, source: Optional[str] = None) -> list[dict[str, Any]]:
    path = Path(path)
    source_name = source or path.stem
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw if isinstance(raw, list) else raw.get("records", [])
    else:
        raise ValueError(f"unsupported concept file: {path}")
    return [
        normalize_record(row, source=source_name)
        for row in rows
        if isinstance(row, Mapping)
    ]


def write_jsonl(records: Iterable[Mapping[str, Any]], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="JSON/JSONL/CSV input; repeat to merge")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for filename in args.input:
        for row in read_records(filename):
            if row["key"] not in seen:
                seen.add(row["key"])
                rows.append(row)
    write_jsonl(rows, args.out)
    print(json.dumps({"records": len(rows), "out": args.out}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
