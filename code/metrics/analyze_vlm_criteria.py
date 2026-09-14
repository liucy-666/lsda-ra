"""VLM binding drift under three criteria, for the three arms (honest reporting).

Criteria:
  c1 rater-strict   : rater fails if either side < 0.5 ; sample fails only if BOTH raters fail
  c2 side-consensus : sample fails only if BOTH raters fail the SAME side
  c3 strict-both    : sample correct only if BOTH raters pass BOTH sides
Arms: SS (native), LSDA (unified knowledge injection), Binding (attention binding v2).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(p: Path):
    d = {}
    if p and Path(p).exists():
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                d[r["id"]] = r
    return d


def sides(row):
    out = []
    for r in ("gpt54", "gemini35"):
        d = (row or {}).get(r) or {}
        if not isinstance(d, dict):
            return None
        rv, lv = d.get("right_is_b"), d.get("left_is_a")
        if not isinstance(rv, (int, float)) and isinstance(lv, (int, float)):
            rv = 1.0 - lv
        if not isinstance(rv, (int, float)) or not isinstance(lv, (int, float)):
            return None
        out.append((float(lv), float(rv)))
    return out if len(out) == 2 else None


def c1(row):
    s = sides(row)
    return None if not s else (not all(lv < 0.5 or rv < 0.5 for lv, rv in s))


def c2(row):
    s = sides(row)
    if not s:
        return None
    return not (all(rv < 0.5 for _, rv in s) or all(lv < 0.5 for lv, _ in s))


def c3(row):
    s = sides(row)
    return None if not s else all(lv >= 0.5 and rv >= 0.5 for lv, rv in s)


CRITERIA = {"c1_rater_strict": c1, "c2_side_consensus": c2, "c3_strict_both": c3}
ARMS = ("SS", "LSDA", "Binding")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", type=Path, required=True)
    ap.add_argument("--census-scores", type=Path, required=True)
    ap.add_argument("--repair-scores", type=Path, nargs="+", required=True)
    ap.add_argument("--binding-scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figure", type=Path, required=True)
    args = ap.parse_args()

    cls = json.loads(args.classification.read_text(encoding="utf-8"))
    census = load(args.census_scores)
    rep = {}
    for p in args.repair_scores:
        rep.update(load(p))
    bind = load(args.binding_scores)

    table = {c: {a: [0, 0] for a in ARMS} for c in CRITERIA}
    for rec in cls.values():
        label = rec.get("label")
        if label not in ("KA", "ME", "BC"):
            continue
        p, s = int(rec["pair"]), int(rec["seed"])
        ss = census.get(f"cen_p{p:03d}_s{s}_SS")
        if label == "ME":
            ls = rep.get(f"lsda_meattrs_p{p:03d}_s{s}")
        elif label == "KA":
            ls = rep.get(f"lsda_ka_p{p:03d}_s{s}")
        else:
            ls = ss
        bd = bind.get(f"bind_p{p:03d}_s{s}")
        for cname, fn in CRITERIA.items():
            for arm, row in (("SS", ss), ("LSDA", ls), ("Binding", bd)):
                v = fn(row)
                if v is not None:
                    table[cname][arm][1] += 1
                    table[cname][arm][0] += 1 if v else 0

    out = {c: {a: {"n": n, "success": k, "drift": (1 - k / n) if n else None} for a, (k, n) in per.items()}
           for c, per in table.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from palette import ARM_COLORS, EDGE, style_axes

    crit = list(CRITERIA)
    x = np.arange(len(crit))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=200)
    for j, a in enumerate(ARMS):
        vals = [(out[c][a]["drift"] or 0) * 100 for c in crit]
        bars = ax.bar(x + (j - 1) * width, vals, width, label=a, color=ARM_COLORS[a], edgecolor=EDGE, linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", fontsize=8, color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels(["c1 rater-strict", "c2 side-consensus", "c3 strict-both"])
    ax.set_ylabel("VLM drift / mis-binding rate (%)")
    ax.set_title("VLM drift under three criteria (n=300)")
    ax.set_ylim(0, 105)
    style_axes(ax)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, bbox_inches="tight")
    print(f"saved {args.figure}")


if __name__ == "__main__":
    main()
