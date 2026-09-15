"""EX6 analysis: repair rate / drift / harm for SS, LL, LSDA-Long, LSDA-attrs.

Primary: repair rate among native-SS failures (same denominator), with pair-clustered bootstrap.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def rows(path: Path) -> dict:
    d = {}
    if path and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                d[r["id"]] = r
    return d


def val(r, rater):
    g = (r or {}).get(rater) or {}
    lv, rv = g.get("left_is_a"), g.get("right_is_b")
    if lv is None and rv is not None:
        lv = 1 - rv
    return (lv, rv) if isinstance(lv, (int, float)) and isinstance(rv, (int, float)) else None


def c1(r):
    a, b = val(r, "gpt54"), val(r, "gemini35")
    if not a or not b:
        return None
    return not ((a[0] < .5 or a[1] < .5) and (b[0] < .5 or b[1] < .5))


def c2(r):
    a, b = val(r, "gpt54"), val(r, "gemini35")
    if not a or not b:
        return None
    return not ((a[0] < .5 and b[0] < .5) or (a[1] < .5 and b[1] < .5))


def c3(r):
    a, b = val(r, "gpt54"), val(r, "gemini35")
    if not a or not b:
        return None
    return a[0] >= .5 and a[1] >= .5 and b[0] >= .5 and b[1] >= .5


CRIT = {"c1": c1, "c2": c2, "c3": c3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", type=Path, required=True)
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--repair", type=Path, required=True)
    ap.add_argument("--meattrs", type=Path, required=True)
    ap.add_argument("--ll", type=Path, required=True)
    ap.add_argument("--lsdalong", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figure", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cls = json.loads(args.classification.read_text(encoding="utf-8"))
    census = rows(args.census)
    rep = rows(args.repair)
    meattrs = rows(args.meattrs)
    ll = rows(args.ll)
    long = rows(args.lsdalong)

    samples = []
    for rec in cls.values():
        if rec.get("label") not in ("KA", "ME", "BC"):
            continue
        i, s, lab = int(rec["pair"]), int(rec["seed"]), rec["label"]
        rid = f"p{i:03d}_s{s}"
        ss = census.get(f"cen_{rid}_SS")
        if lab == "ME":
            lsda_attrs = meattrs.get(f"lsda_meattrs_{rid}")
        elif lab == "KA":
            lsda_attrs = rep.get(f"lsda_ka_{rid}")
        else:
            lsda_attrs = ss
        samples.append({
            "pair": i, "seed": s, "label": lab,
            "SS": ss, "LSDA-attrs": lsda_attrs,
            "LSDA-Long": long.get(f"lsdalong_{rid}"),
            "LL": ll.get(f"ll_{rid}"),
        })

    arms = ["SS", "LSDA-attrs", "LSDA-Long", "LL"]
    res = {}
    for cname, fn in CRIT.items():
        ok = {a: {i: fn(samples[i][a]) for i in range(len(samples))} for a in arms}
        per = {}
        for a in arms[1:]:
            idx = [i for i in range(len(samples)) if ok["SS"][i] is False and ok[a][i] is not None]
            repaired = sum(1 for i in idx if ok[a][i] is True)
            nfail = len(idx)
            base_ok = [i for i in range(len(samples)) if ok["SS"][i] is True and ok[a][i] is not None]
            harm = sum(1 for i in base_ok if ok[a][i] is False)
            allv = [i for i in range(len(samples)) if ok[a][i] is not None]
            succ = sum(1 for i in allv if ok[a][i])
            per[a] = {"n": len(allv), "success": succ, "drift": 1 - succ / len(allv) if allv else None,
                      "ss_fail": nfail, "repaired": repaired,
                      "repair_rate": repaired / nfail if nfail else None,
                      "harm": harm, "harm_rate": harm / len(base_ok) if base_ok else None}
        # bootstrap repair-rate differences by pair
        by_pair = defaultdict(list)
        for i, sm in enumerate(samples):
            by_pair[sm["pair"]].append(i)
        pair_keys = list(by_pair)
        rng = random.Random(args.seed)
        diffs = {f"{a}-LL": [] for a in ("LSDA-attrs", "LSDA-Long")}
        diffs["LSDA-Long-LSDA-attrs"] = []
        for _ in range(args.boot):
            idx = []
            for _ in range(len(pair_keys)):
                idx += by_pair[rng.choice(pair_keys)]
            fail = [i for i in idx if ok["SS"][i] is False]
            if not fail:
                continue
            rr = {a: (sum(1 for i in fail if ok[a][i] is True) / len(fail)) for a in arms[1:]}
            diffs["LSDA-attrs-LL"].append(rr["LSDA-attrs"] - rr["LL"])
            diffs["LSDA-Long-LL"].append(rr["LSDA-Long"] - rr["LL"])
            diffs["LSDA-Long-LSDA-attrs"].append(rr["LSDA-Long"] - rr["LSDA-attrs"])

        def ci(v):
            if not v:
                return [None, None]
            v = sorted(v)
            return [v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]]

        per["bootstrap"] = {k: {"mean": (sum(v) / len(v) if v else None), "ci95": ci(v)} for k, v in diffs.items()}
        res[cname] = per

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    # figure: repair rate c1
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_arms = ["LSDA-attrs", "LSDA-Long", "LL"]
    vals = [100 * (res["c1"][a]["repair_rate"] or 0) for a in plot_arms]
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=160)
    bars = ax.bar(plot_arms, vals, color=["#c98b48", "#2f9e6e", "#4a6fa5"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.0, f"{v:.1f}", ha="center", fontsize=10)
    ax.set_ylabel("Repair rate among native SS failures (%)")
    ax.set_title("Injection location: c1 repair rate (100 pairs x 3 seeds)")
    ax.set_ylim(0, max(vals) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print("saved", args.out, args.figure)


if __name__ == "__main__":
    main()
