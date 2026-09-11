"""Select qualifying seeds for the causal experiment (strict inclusion criterion).

Qualifying: standalone A correct AND standalone B correct (both raters >= thr)
AND mixed SS fails (both raters < thr on right_is_b).
"""
import argparse
import json
import re
from pathlib import Path


def rater_values(rec: dict, key: str) -> list:
    """All numeric values for `key` across rater sub-dicts."""
    vals = []
    for k, v in rec.items():
        if k in ("id", "image"):
            continue
        if isinstance(v, dict):
            x = v.get(key)
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                vals.append(float(x))
    return vals


def binding_values(rec: dict) -> list:
    """right_is_b, falling back to 1 - right_is_a when the rater swapped the key."""
    vals = []
    for k, v in rec.items():
        if k in ("id", "image") or not isinstance(v, dict):
            continue
        b = v.get("right_is_b")
        if isinstance(b, (int, float)) and not isinstance(b, bool):
            vals.append(float(b))
            continue
        a = v.get("right_is_a")
        if isinstance(a, (int, float)) and not isinstance(a, bool):
            vals.append(1.0 - float(a))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    rows = {}
    for line in args.scores.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r

    grouped: dict = {}
    for rid, r in rows.items():
        m = re.match(r"e0_p(\d+)_s(\d+)_(A|B|SS)$", rid)
        if not m:
            continue
        grouped.setdefault((int(m.group(1)), int(m.group(2))), {})[m.group(3)] = r

    detail, qualified = [], []
    for (pair, seed), conds in sorted(grouped.items()):
        if not all(c in conds for c in ("A", "B", "SS")):
            continue
        a, b, ss = conds["A"], conds["B"], conds["SS"]
        a_vals = rater_values(a, "matches")
        b_vals = rater_values(b, "matches")
        ss_vals = binding_values(ss)
        a_ok = len(a_vals) > 0 and all(v >= args.thr for v in a_vals)
        b_ok = len(b_vals) > 0 and all(v >= args.thr for v in b_vals)
        ss_fail = len(ss_vals) > 0 and all(v < args.thr for v in ss_vals)
        row = {
            "pair": pair, "seed": seed, "a_ok": a_ok, "b_ok": b_ok, "ss_fail": ss_fail,
            "a_vals": a_vals, "b_vals": b_vals, "ss_vals": ss_vals,
            "struct_vals": rater_values(ss, "structure"),
        }
        detail.append(row)
        if a_ok and b_ok and ss_fail:
            qualified.append(seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"qualified": qualified, "detail": detail}, indent=1), encoding="utf-8")
    print(json.dumps({"qualified": qualified, "n": len(qualified)}))


if __name__ == "__main__":
    main()
