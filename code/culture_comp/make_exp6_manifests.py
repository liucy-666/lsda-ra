"""Build EXP_6 job manifests: LL (global long prompt) and LSDA-Long (per-expert long prompt).

LL       : one image per (pair, seed), prompt = 组合Prompt LL.
LSDA-Long: one LSDA job per failure sample (KA/ME), a_prompt=long_A, b_prompt=long_B, ss_prompt=SS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--classification", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seeds", default="1011,1012,1013")
    args = ap.parse_args()

    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))
    cls = json.loads(args.classification.read_text(encoding="utf-8"))
    seeds = [int(s) for s in args.seeds.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ll_jobs, long_jobs = [], []
    for i, rec in enumerate(pairs, start=1):
        ll = rec["组合Prompt LL"]
        for s in seeds:
            ll_jobs.append({"run_id": f"ll_p{i:03d}_s{s}", "pair": i, "seed": s, "prompt": ll})

    for v in cls.values():
        if v.get("label") not in ("KA", "ME"):
            continue
        i, s = int(v["pair"]), int(v["seed"])
        rec = pairs[i - 1]
        long_jobs.append({
            "run_id": f"lsdalong_p{i:03d}_s{s}", "pair": i, "seed": s,
            "a_prompt": rec["文化物体A的长文本描述"],
            "b_prompt": rec["文化物体B的长文本描述"],
            "ss_prompt": rec["组合Prompt SS"], "kind": v["label"],
        })

    (args.out_dir / "ll_jobs.json").write_text(json.dumps(ll_jobs, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out_dir / "lsdalong_jobs.json").write_text(json.dumps(long_jobs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"ll_jobs": len(ll_jobs), "lsdalong_jobs": len(long_jobs)}))


if __name__ == "__main__":
    main()
