"""Merge pilot scores (score_pilot_c1c2.py output) + sidecar cost into the
overlay format expected by h2_timing_analysis.py --scores.

binding_score per branch semantics: target=A -> binding_A; target=B -> binding_B;
target=AB -> mean(binding_A, binding_B). structure_score = mean structure.
compute_seconds from sidecar "seconds". Leakage/artifact derived from scores.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--scores", type=Path, required=True)
ap.add_argument("--sidecars-root", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

sidecars = {}
for p in Path(args.sidecars_root).rglob("*.json"):
    try:
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("replay_id"):
            sidecars[r["replay_id"]] = r
    except Exception:
        pass

lines = []
for line in Path(args.scores).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    rid = row["replay_id"]
    if "error" in row.get("QWEN", {}) or "error" in row.get("GEMINI", {}):
        lines.append({"replay_id": rid, "status": "scorer_error"})
        continue
    target = row["target"]
    if target == "A":
        binding = row["binding_A"]
    elif target == "B":
        binding = row["binding_B"]
    else:
        binding = (row["binding_A"] + row["binding_B"]) / 2
    sc = sidecars.get(rid, {})
    lines.append({
        "replay_id": rid,
        "status": "generated",
        "binding_score": binding,
        "structure_score": row["structure"],
        "leakage_score": 1.0 - min(row["binding_A"], row["binding_B"]),
        "artifact_score": 0.0 if "none" in row.get("QWEN", {}).get("artifacts", []) else 1.0,
        "compute_seconds": sc.get("seconds"),
    })
Path(args.out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")
print(f"merged {len(lines)} rows -> {args.out}")
