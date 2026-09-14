"""Per-class analysis with unambiguous denominators.

Failure classes (ME, KA): denominator = samples of that class that FAILED NATIVELY
(c1 on the native SS). Then SS = 100% by construction, and LSDA / Binding report the
residual failure rate among native failures.

Correct class (BC): report the *damage rate* (fraction of natively-correct samples that the
method turns into failures).
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(p):
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

    # native-failure subsets for ME/KA ; natively-correct subset for BC damage
    fail_sub = {"ME": [], "KA": []}
    bc_correct = []
    for rec in cls.values():
        label = rec.get("label")
        p, s = int(rec["pair"]), int(rec["seed"])
        ss = census.get(f"cen_p{p:03d}_s{s}_SS")
        if label in ("ME", "KA"):
            ls = rep.get(f"lsda_meattrs_p{p:03d}_s{s}") if label == "ME" else rep.get(f"lsda_ka_p{p:03d}_s{s}")
            bd = bind.get(f"bind_p{p:03d}_s{s}")
            if c1(ss) is False:          # native failure
                fail_sub[label].append((ls, bd))
        elif label == "BC":
            bd = bind.get(f"bind_p{p:03d}_s{s}")
            if c1(ss) is not False:      # natively correct
                bc_correct.append((bd,))

    def resid(items):
        out = {"n": 0, "LSDA": 0, "Binding": 0}
        for ls, bd in items:
            out["n"] += 1
            out["LSDA"] += 1 if c1(ls) is False else 0
            out["Binding"] += 1 if c1(bd) is False else 0
        return out

    rows = {}
    for lab in ("ME", "KA"):
        r = resid(fail_sub[lab])
        n = r["n"]
        rows[lab] = {
            "n_native_fail": n,
            "SS": 1.0,  # by construction
            "LSDA": (r["LSDA"] / n) if n else None,
            "Binding": (r["Binding"] / n) if n else None,
        }
    # BC damage
    nb = len(bc_correct)
    bc_dmg_lsda = 1.0  # LSDA does not modify BC samples (uses native)
    rows["BC"] = {
        "n_correct": nb,
        "LSDA_damage": 0.0,
        "Binding_damage": (sum(1 for (bd,) in bc_correct if c1(bd) is False) / nb) if nb else None,
    }
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from palette import ARM_COLORS, EDGE, style_axes

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), dpi=200, gridspec_kw={"width_ratios": [2.1, 1]})
    # Panel A: residual failure among native failures
    ax = axes[0]
    labs = ["ME", "KA"]
    x = np.arange(len(labs))
    width = 0.34
    for j, a in enumerate(("LSDA", "Binding")):
        vals = [(rows[l][a] or 0) * 100 for l in labs]
        bars = ax.bar(x + (j - 0.5) * width, vals, width, label=a, color=ARM_COLORS[a], edgecolor=EDGE, linewidth=0.7)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=9, color="#222222")
    ax.axhline(100, color="#777777", linestyle="--", linewidth=1.0)
    ax.text(1.45, 103, "Native SS = 100% (by construction)", fontsize=8, color="#555555", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={rows[l]['n_native_fail']})" for l in labs])
    ax.set_ylabel("Residual failure rate (%)")
    ax.set_title("Among samples that failed natively")
    ax.set_ylim(0, 115)
    style_axes(ax)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    # Panel B: damage on correct samples
    ax = axes[1]
    x = np.arange(1)
    for j, (a, key) in enumerate((("LSDA", "LSDA_damage"), ("Binding", "Binding_damage"))):
        v = (rows["BC"][key] or 0) * 100
        bar = ax.bar(x + (j - 0.5) * width, [v], width, label=a, color=ARM_COLORS[a], edgecolor=EDGE, linewidth=0.7)
        ax.text(bar[0].get_x() + bar[0].get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", fontsize=9, color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels([f"BC (n={rows['BC']['n_correct']})"])
    ax.set_ylabel("Damage rate (%)")
    ax.set_title("Damage on natively correct")
    ax.set_ylim(0, 40)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(args.figure, bbox_inches="tight")
    print(f"saved {args.figure}")


if __name__ == "__main__":
    main()
