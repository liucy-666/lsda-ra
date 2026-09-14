"""Classify census samples into KA / ME / BC from dual-VLM scores (strict dual).

Input: score_images.py JSONL with ids cen_p{pair}_s{seed}_{A|B|SS}.
Rule:
  a_ok/b_ok = both raters' `matches` >= 0.5
  leaked    = both raters fail the SAME side (< 0.5) on SS
  KA = not a_ok or not b_ok ; ME = a_ok and b_ok and leaked ; BC = a_ok and b_ok and not leaked
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

THR = 0.5


def num(row, key):
    v = row.get(key) if isinstance(row, dict) else None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def binding(row):
    b = num(row, "right_is_b")
    if b is not None:
        return b
    a = num(row, "left_is_a")
    return 1.0 - a if a is not None else None


def left_ok(row):
    return num(row, "left_is_a")


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

    grouped = defaultdict(dict)
    for rid, r in rows.items():
        m = re.match(r"cen_p(\d+)_s(\d+)_(A|B|SS)$", rid)
        if not m:
            continue
        raters = {k: v for k, v in r.items() if k not in ("id", "image") and isinstance(v, dict)}
        grouped[(int(m.group(1)), int(m.group(2)))][m.group(3)] = raters

    out = {}
    counts = {"KA": 0, "ME": 0, "BC": 0, "unknown": 0}
    for (pair, seed), kinds in sorted(grouped.items()):
        a, b, ss = kinds.get("A"), kinds.get("B"), kinds.get("SS")
        a_vals = [num(v, "matches") for v in (a or {}).values()]
        b_vals = [num(v, "matches") for v in (b or {}).values()]
        a_ok = len(a_vals) >= 2 and all(v is not None and v >= THR for v in a_vals)
        b_ok = len(b_vals) >= 2 and all(v is not None and v >= THR for v in b_vals)
        ss_rows = list((ss or {}).values())
        right = [binding(v) for v in ss_rows]
        left = [left_ok(v) for v in ss_rows]
        right_fail = len(right) >= 2 and all(v is not None and v < THR for v in right)
        left_fail = len(left) >= 2 and all(v is not None and v < THR for v in left)
        leaked = right_fail or left_fail
        if not (len(a_vals) >= 2 and len(b_vals) >= 2 and len(ss_rows) >= 2):
            label = "unknown"
        elif not a_ok or not b_ok:
            label = "KA"
        elif leaked:
            label = "ME"
        else:
            label = "BC"
        counts[label] += 1
        out[f"{pair}|{seed}"] = {
            "pair": pair, "seed": seed, "label": label,
            "a_ok": a_ok, "b_ok": b_ok, "leaked": leaked,
            "right_fail": right_fail, "left_fail": left_fail,
            "a_vals": a_vals, "b_vals": b_vals, "right_vals": right, "left_vals": left,
            "struct_ss": [num(v, "structure") for v in ss_rows],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"total": len(out), **counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
