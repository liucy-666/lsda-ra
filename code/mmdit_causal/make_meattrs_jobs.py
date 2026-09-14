"""Knowledge-uniform arm: apply rule-extracted attribute phrases to ALL failures.

KA failures reuse the routed (ka) runs; ME failures need new 'meattrs' runs so the fourth
bar ('knowledge applied to every failure, no diagnosis') can be scored.
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
        if rec.get("label") != "ME":
            continue
        pair, seed = int(rec["pair"]), int(rec["seed"])
        a, b, ss = a_only_prompt(pair), b_only_prompt(pair), ss_prompt(pair)
        at = attrs[f"{pair:03d}"]
        a_r = f"{a}, {at['A']['attrs']}" if at["A"]["attrs"] else a
        b_r = f"{b}, {at['B']['attrs']}" if at["B"]["attrs"] else b
        jobs.append({
            "run_id": f"lsda_meattrs_p{pair:03d}_s{seed}", "pair": pair, "seed": seed,
            "a_prompt": a_r, "b_prompt": b_r, "ss_prompt": ss, "kind": "meattrs",
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(jobs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"meattrs_jobs": len(jobs)}))


if __name__ == "__main__":
    main()
