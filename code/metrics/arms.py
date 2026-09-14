"""Shared helpers: resolve the three evaluation arms for each census sample.

Arms:
  SS      : native composition (census)
  LSDA    : unified knowledge injection (ME -> meattrs, KA -> ka, BC -> native)
  Binding : attention-binding baseline (binding2)
"""
from __future__ import annotations

import json
from pathlib import Path

ARMS = ("SS", "LSDA", "Binding")


def load_classification(path: Path) -> list[dict]:
    cls = json.loads(path.read_text(encoding="utf-8"))
    rows = [v for v in cls.values() if v.get("label") in ("KA", "ME", "BC")]
    rows.sort(key=lambda r: (int(r["pair"]), int(r["seed"])))
    return rows


def resolve(rec: dict, census_dir: Path, lsda_dir: Path, binding_dir: Path) -> dict:
    pair, seed, label = int(rec["pair"]), int(rec["seed"]), rec["label"]
    ss = census_dir / f"pair{pair:03d}_seed{seed}_SS.jpg"
    if label == "ME":
        lsda = lsda_dir / f"lsda_meattrs_p{pair:03d}_s{seed}.jpg"
    elif label == "KA":
        lsda = lsda_dir / f"lsda_ka_p{pair:03d}_s{seed}.jpg"
    else:
        lsda = ss
    binding = binding_dir / f"binding2_p{pair:03d}_s{seed}.jpg"
    return {"SS": ss, "LSDA": lsda, "Binding": binding}


def refs_for(rec: dict, census_dir: Path) -> tuple[Path, Path]:
    pair, seed = int(rec["pair"]), int(rec["seed"])
    return (
        census_dir / f"pair{pair:03d}_seed{seed}_A.jpg",
        census_dir / f"pair{pair:03d}_seed{seed}_B.jpg",
    )
