"""Direction-aware mixed/LSDA/donor value projection analysis.

The helper consumes saved vectors from an activation capture. It is deliberately
model-independent so the same audit can be rerun after a GPU job without loading
the diffusion checkpoint on the analysis workstation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np


def _vector(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("mechanism vectors must not be empty")
    return array


def value_correction_metrics(mixed: Any, lsda: Any, donor: Any, eps: float = 1e-12) -> dict[str, float]:
    """Measure whether the LSDA correction points toward the donor value."""
    mixed_v, lsda_v, donor_v = map(_vector, (mixed, lsda, donor))
    if not (mixed_v.shape == lsda_v.shape == donor_v.shape):
        raise ValueError("mixed, lsda, and donor vectors must have the same shape")
    direction = donor_v - mixed_v
    correction = lsda_v - mixed_v
    denom = float(np.dot(direction, direction))
    before = float(np.linalg.norm(mixed_v - donor_v))
    if denom <= eps:
        return {
            "projection": 0.0,
            "cosine": 0.0,
            "distance_before": before,
            "distance_after": float(np.linalg.norm(lsda_v - donor_v)),
            "distance_reduction": before - float(np.linalg.norm(lsda_v - donor_v)),
            "degenerate_direction": 1.0,
        }
    correction_norm = float(np.linalg.norm(correction))
    projection = float(np.dot(correction, direction) / denom)
    cosine = float(np.dot(correction, direction) / max(eps, correction_norm * np.sqrt(denom))) if correction_norm else 0.0
    after = float(np.linalg.norm(lsda_v - donor_v))
    return {
        "projection": projection,
        "cosine": cosine,
        "distance_before": before,
        "distance_after": after,
        "distance_reduction": before - after,
        "degenerate_direction": 0.0,
    }


def analyze_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        metrics = value_correction_metrics(row["mixed"], row["lsda"], row["donor"])
        enriched = dict(row)
        enriched["metrics"] = metrics
        enriched["toward_donor"] = metrics["projection"] > 0 and metrics["distance_reduction"] > 0
        output.append(enriched)
    return output


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL captures with mixed/lsda/donor arrays")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = analyze_rows(rows)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in result:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(result), "toward_donor": sum(row["toward_donor"] for row in result), "out": str(destination)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
