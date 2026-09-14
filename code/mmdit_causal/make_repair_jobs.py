"""Build LSDA repair jobs for the KA/ME census.

Arms:
  ME sample      -> one job with short prompts (routed == uniform for ME)
  KA sample      -> 'KA' job with rule-extracted attrs appended to the failing entity's prompt
                    + 'uniform' job with short prompts (control: LSDA without knowledge)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pairs100 import a_only_prompt, b_only_prompt, ss_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", type=Path, required=True)
    ap.add_argument("--attrs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cls = json.loads(args.classification.read_text(encoding="utf-8"))
    attrs = json.loads(args.attrs.read_text(encoding="utf-8"))

    jobs = []
    for rec in cls.values():
        if rec.get("label") not in ("KA", "ME"):
            continue
        pair, seed = int(rec["pair"]), int(rec["seed"])
        a, b, ss = a_only_prompt(pair), b_only_prompt(pair), ss_prompt(pair)
        at = attrs[f"{pair:03d}"]
        if rec["label"] == "ME":
            jobs.append({
                "run_id": f"lsda_me_p{pair:03d}_s{seed}", "pair": pair, "seed": seed,
                "a_prompt": a, "b_prompt": b, "ss_prompt": ss, "kind": "ME",
            })
        else:
            a_fail, b_fail = not rec["a_ok"], not rec["b_ok"]
            a_r = f"{a}, {at['A']['attrs']}" if a_fail and at["A"]["attrs"] else a
            b_r = f"{b}, {at['B']['attrs']}" if b_fail and at["B"]["attrs"] else b
            jobs.append({
                "run_id": f"lsda_ka_p{pair:03d}_s{seed}", "pair": pair, "seed": seed,
                "a_prompt": a_r, "b_prompt": b_r, "ss_prompt": ss, "kind": "KA",
            })
            jobs.append({
                "run_id": f"lsda_uniform_p{pair:03d}_s{seed}", "pair": pair, "seed": seed,
                "a_prompt": a, "b_prompt": b, "ss_prompt": ss, "kind": "uniform",
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(jobs, ensure_ascii=False, indent=1), encoding="utf-8")
    kinds = {}
    for j in jobs:
        kinds[j["kind"]] = kinds.get(j["kind"], 0) + 1
    print(json.dumps({"jobs": len(jobs), **kinds}, ensure_ascii=False))


if __name__ == "__main__":
    main()
