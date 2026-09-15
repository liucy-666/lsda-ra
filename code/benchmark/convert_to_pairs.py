"""Convert composed KB prompts to the existing frozen generation schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional


def _article(text: str) -> str:
    return "an" if text[:1].lower() in "aeiou" else "a"


def _entity(row: Mapping[str, Any]) -> dict[str, Any]:
    concept = str(row.get("concept", "")).strip()
    object_type = str(row.get("object_type") or concept).strip()
    knowledge = str(row.get("knowledge_text", "")).strip()
    return {
        "name": concept,
        "noun": object_type,
        "tradition": row.get("culture"),
        "attr": knowledge or concept,
        "knowledge_text": knowledge,
    }


def _phrase(entity: Mapping[str, Any]) -> str:
    name = str(entity["name"]).strip()
    lower = name.lower()
    prefix = "" if lower.startswith(("a ", "an ", "the ")) else _article(name) + " "
    return f"{prefix}{name} with {entity['attr']}"


def _seed(pair_id: str, kind: str, index: int) -> int:
    digest = hashlib.sha256(f"{pair_id}|{kind}|{index}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def convert(
    rows: list[Mapping[str, Any]], *, replicates: int = 3, images_per_seed: int = 3
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        left, right = _entity(row["a"]), _entity(row["b"])
        pair_id = str(row.get("pair_id"))
        combo = f"{_phrase(left).capitalize()} on the left and {_phrase(right)} on the right."

        def solo(entity: Mapping[str, Any]) -> str:
            return _phrase(entity).capitalize() + "."

        output.append(
            {
                "pid": pair_id,
                "a_id": str(row["a"].get("key") or row["a"].get("concept")),
                "b_id": str(row["b"].get("key") or row["b"].get("concept")),
                "domain": row.get("category"),
                "contrast": "cross",
                "same_noun": left["noun"].lower() == right["noun"].lower(),
                "a": left,
                "b": right,
                "prompts": {"solo_a": solo(left), "solo_b": solo(right), "combo": combo},
                "seeds": {
                    kind: [_seed(pair_id, kind, index) for index in range(replicates)]
                    for kind in ("solo_a", "solo_b", "combo")
                },
                "source_pair": dict(row),
                "images_per_seed": int(images_per_seed),
            }
        )
    return output


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--images-per-seed", type=int, default=3)
    args = parser.parse_args(argv)
    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = convert(rows, replicates=args.replicates, images_per_seed=args.images_per_seed)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": len(result), "out": str(destination)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
