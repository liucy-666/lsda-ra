"""Build the unified evaluation index for the automated LSDA assessment.

Inputs:
  - experiment/2026_8_25_EXP_1/.../binary_vqa_v2/key/blind_map.json
  - data/SS|LSDA|Standalone_A|Standalone_B/2026_8_25_EXP_1/sample_XXXX.jpg
  - binary_vqa_v2/ratings/{QWEN,GEMINI}/ratings_*.jsonl (existing VLM ratings)

Output:
  - experiment/2026_8_31_EXP_2/manifests/eval_index.json (one row per eval_id)
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"D:\Python\MMDIT")
EXP = ROOT / "experiment" / "2026_8_31_EXP_2"
VQA_ROOT = (
    ROOT
    / "experiment"
    / "2026_8_25_EXP_1"
    / "cultural100_records"
    / "experiment_4500"
    / "binary_vqa_v2"
)
IMG = {
    "native_SS": ROOT / "data" / "SS" / "2026_8_25_EXP_1",
    "lsda_clean": ROOT / "data" / "LSDA" / "2026_8_25_EXP_1",
    "standalone_A": ROOT / "data" / "Standalone_A" / "2026_8_25_EXP_1",
    "standalone_B": ROOT / "data" / "Standalone_B" / "2026_8_25_EXP_1",
}


def load_ratings(rater: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted((VQA_ROOT / "ratings" / rater).glob("ratings_*.jsonl")):
        for line in p.read_text(encoding="utf-8-sig").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("eval_id") and r["eval_id"] not in out:
                out[r["eval_id"]] = r
    return out


def main() -> None:
    blind = json.loads((VQA_ROOT / "key" / "blind_map.json").read_text(encoding="utf-8"))
    qwen = load_ratings("QWEN")
    gemini = load_ratings("GEMINI")
    print(f"blind_map entries: {len(blind)}, Qwen ratings: {len(qwen)}, Gemini ratings: {len(gemini)}")

    rows = []
    missing_files = []
    for b in blind:
        eid = b["eval_id"]
        sid = b["sample_id"]
        row = {
            "eval_id": eid,
            "sample_id": sid,
            "pair_id": b["pair_id"],
            "pair_index": b["pair_index"],
            "seed_group": b["seed_group"],
            "replicate": b["replicate"],
            "latent_seed": b["latent_seed"],
            "condition": b["condition"],
            "entity_A": b["entity_A"],
            "entity_B": b["entity_B"],
            "entity_A_diagnostic": b.get("entity_A_diagnostic", ""),
            "entity_B_diagnostic": b.get("entity_B_diagnostic", ""),
        }
        # resolve image paths
        paths = {}
        for cond, d in IMG.items():
            p = d / f"{sid}.jpg"
            paths[cond] = str(p)
            if not p.exists():
                missing_files.append(str(p))
        row["images"] = paths
        # existing VLM ratings for this eval_id
        q = qwen.get(eid)
        g = gemini.get(eid)
        row["qwen"] = None if not q else {
            "left_choice": q.get("left_choice"),
            "right_choice": q.get("right_choice"),
            "correct_binding": bool(q.get("correct_binding")),
            "model": q.get("model"),
        }
        row["gemini"] = None if not g else {
            "left_choice": g.get("left_choice"),
            "right_choice": g.get("right_choice"),
            "correct_binding": bool(g.get("correct_binding")),
            "model": g.get("model"),
        }
        rows.append(row)

    (EXP / "manifests").mkdir(parents=True, exist_ok=True)
    (EXP / "manifests" / "eval_index.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"rows written: {len(rows)}")
    print(f"missing image files: {len(missing_files)}")
    for m in missing_files[:10]:
        print("  MISSING:", m)

    # completeness by condition
    from collections import Counter

    c = Counter(r["condition"] for r in rows)
    print("conditions:", dict(c))
    qw_ok = sum(1 for r in rows if r["qwen"])
    gm_ok = sum(1 for r in rows if r["gemini"])
    print(f"with Qwen rating: {qw_ok}, with Gemini rating: {gm_ok}")


if __name__ == "__main__":
    main()
