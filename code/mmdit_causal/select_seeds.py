"""Select qualifying seeds for the causal experiment (strict inclusion criterion).

Qualifying: standalone A correct AND standalone B correct (both raters >= thr)
AND mixed SS fails (both raters < thr on right_is_b).
"""
import argparse
import json
import re
from pathlib import Path


def get(rater_row: dict, key: str):
    if not isinstance(rater_row, dict):
        return None
    v = rater_row.get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


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
        aq, ag = get(a.get("qwen"), "matches"), get(a.get("gemini"), "matches")
        bq, bg = get(b.get("qwen"), "matches"), get(b.get("gemini"), "matches")
        sq, sg = get(ss.get("qwen"), "right_is_b"), get(ss.get("gemini"), "right_is_b")
        a_ok = aq is not None and ag is not None and aq >= args.thr and ag >= args.thr
        b_ok = bq is not None and bg is not None and bq >= args.thr and bg >= args.thr
        ss_fail = sq is not None and sg is not None and sq < args.thr and sg < args.thr
        row = {
            "pair": pair, "seed": seed, "a_ok": a_ok, "b_ok": b_ok, "ss_fail": ss_fail,
            "ss_qwen": sq, "ss_gemini": sg,
            "struct_qwen": get(ss.get("qwen"), "structure"), "struct_gemini": get(ss.get("gemini"), "structure"),
        }
        detail.append(row)
        if a_ok and b_ok and ss_fail:
            qualified.append(seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"qualified": qualified, "detail": detail}, indent=1), encoding="utf-8")
    print(json.dumps({"qualified": qualified, "n": len(qualified)}))


if __name__ == "__main__":
    main()
