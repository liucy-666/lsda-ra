"""Analyze the deep-dive scores: strict dual-judge leakage + inclusion criterion."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def binding_of(row):
    if not isinstance(row, dict):
        return None
    v = row.get("right_is_b")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    a = row.get("right_is_a")
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return 1.0 - float(a)
    return None


def matches_of(row):
    if not isinstance(row, dict):
        return None
    v = row.get("matches")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def left_ok(row):
    if not isinstance(row, dict):
        return None
    v = row.get("left_is_a")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = {}
    for line in args.scores.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r

    recs = defaultdict(dict)
    for rid, r in rows.items():
        m = re.match(r"dd_(pair_\d+)_s(\d+)_(SS|A|B)$", rid)
        if not m:
            continue
        pair, seed, kind = m.group(1), int(m.group(2)), m.group(3)
        raters = {k: v for k, v in r.items() if k not in ("id", "image") and isinstance(v, dict)}
        recs[(pair, seed)][kind] = raters

    out = {}
    for (pair, seed), kinds in sorted(recs.items()):
        a, b, ss = kinds.get("A"), kinds.get("B"), kinds.get("SS")
        a_vals = [matches_of(v) for v in (a or {}).values()]
        b_vals = [matches_of(v) for v in (b or {}).values()]
        a_ok = len(a_vals) > 0 and all(v is not None and v >= 0.5 for v in a_vals)
        b_ok = len(b_vals) > 0 and all(v is not None and v >= 0.5 for v in b_vals)
        ss_rows = list((ss or {}).values())
        right_vals = [binding_of(v) for v in ss_rows]
        left_vals = [left_ok(v) for v in ss_rows]
        # strict dual: both raters fail the same side
        right_fail = len(right_vals) >= 2 and all(v is not None and v < 0.5 for v in right_vals)
        left_fail = len(left_vals) >= 2 and all(v is not None and v < 0.5 for v in left_vals)
        leaked = right_fail or left_fail
        out[f"{pair}|{seed}"] = {
            "a_ok": a_ok, "b_ok": b_ok,
            "right_vals": right_vals, "left_vals": left_vals,
            "right_fail": right_fail, "left_fail": left_fail,
            "leaked": leaked, "included": a_ok and b_ok and leaked,
        }

    n_leak = sum(1 for v in out.values() if v["leaked"])
    n_incl = sum(1 for v in out.values() if v["included"])
    print(f"total seeds={len(out)} leaked={n_leak} included(A&B ok + leaked)={n_incl}")
    by_pair = defaultdict(lambda: {"n": 0, "leak": 0, "incl": 0})
    for key, v in out.items():
        p = key.split("|")[0]
        by_pair[p]["n"] += 1
        by_pair[p]["leak"] += int(v["leaked"])
        by_pair[p]["incl"] += int(v["included"])
    for p, e in sorted(by_pair.items()):
        print(f"  {p}: n={e['n']} leaked={e['leak']} included={e['incl']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved", args.out)


if __name__ == "__main__":
    main()
