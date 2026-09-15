"""Assemble EXP_6 final table + bar chart: 5 metrics x 4 arms (SS, LSDA-attrs, LSDA-Long, LL)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(p: Path) -> list:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def vlm_c1(r):
    g = (r or {}).get("gpt54") or {}
    m = (r or {}).get("gemini35") or {}

    def vv(d):
        a, b = d.get("left_is_a"), d.get("right_is_b")
        if a is None and isinstance(b, (int, float)):
            a = 1 - b
        return (a, b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
    a, b = vv(g), vv(m)
    if not a or not b:
        return None
    return not ((a[0] < .5 or a[1] < .5) and (b[0] < .5 or b[1] < .5))


def drift(correct_flags):
    n = sum(1 for c in correct_flags if c is not None)
    ok = sum(1 for c in correct_flags if c is True)
    return {"n": n, "success": ok, "drift": (1 - ok / n) if n else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp3-ratings", type=Path, required=True)
    ap.add_argument("--exp6-ratings", type=Path, required=True)
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--inject3", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figure", type=Path, required=True)
    args = ap.parse_args()

    arms = ["SS", "LSDA-attrs", "LSDA-Long", "LL"]
    table = {m: {a: None for a in arms} for m in ("VLM", "LBP_GLCM", "Gram", "kNN", "SigLIP")}

    # ---- objective: exp3 (SS, LSDA) + exp6 (Binding -> our arms)
    tex = rows(args.exp3_ratings / "metric_texture.jsonl") + rows(args.exp6_ratings / "metric_texture_LL.jsonl") + rows(args.exp6_ratings / "metric_texture_LSDA_Long.jsonl")
    dino = rows(args.exp3_ratings / "metric_dino.jsonl") + rows(args.exp6_ratings / "metric_dino_LL.jsonl") + rows(args.exp6_ratings / "metric_dino_LSDA_Long.jsonl")
    sig = rows(args.exp3_ratings / "metric_siglip.jsonl") + rows(args.exp6_ratings / "metric_siglip_LL.jsonl") + rows(args.exp6_ratings / "metric_siglip_LSDA_Long.jsonl")

    def collect(rs, key, arm):
        flags = []
        for r in rs:
            a = r.get("arm")
            if armmap(a) == arm:
                flags.append(r.get(key))
        return drift(flags)

    armmap = lambda a: {"SS": "SS", "LSDA": "LSDA-attrs", "Binding": a and a or "Binding"}.get(a, a)  # noqa

    def arm_of(r, target):
        a = r.get("arm")
        if target in ("LSDA-Long", "LL"):
            src = "LL" if target == "LL" else "LSDA_Long"
            return a == "Binding" and src in str(r.get("_src", ""))
        return a == {"SS": "SS", "LSDA-attrs": "LSDA"}[target]

    # tag source files explicitly (avoid ambiguity between the two Binding-arm files)
    def tagged(list_of_pairs):
        out = []
        for rs, src in list_of_pairs:
            for r in rs:
                r = dict(r)
                r["_src"] = src
                out.append(r)
        return out

    tex = tagged([(rows(args.exp3_ratings / "metric_texture.jsonl"), "exp3"),
                  (rows(args.exp6_ratings / "metric_texture_LL.jsonl"), "LL"),
                  (rows(args.exp6_ratings / "metric_texture_LSDA_Long.jsonl"), "LSDA_Long")])
    dino = tagged([(rows(args.exp3_ratings / "metric_dino.jsonl"), "exp3"),
                   (rows(args.exp6_ratings / "metric_dino_LL.jsonl"), "LL"),
                   (rows(args.exp6_ratings / "metric_dino_LSDA_Long.jsonl"), "LSDA_Long")])
    sig = tagged([(rows(args.exp3_ratings / "metric_siglip.jsonl"), "exp3"),
                  (rows(args.exp6_ratings / "metric_siglip_LL.jsonl"), "LL"),
                  (rows(args.exp6_ratings / "metric_siglip_LSDA_Long.jsonl"), "LSDA_Long")])

    for target in ("SS", "LSDA-attrs"):
        table["LBP_GLCM"][target] = collect([r for r in tex if r["_src"] == "exp3"], "correct", target)
        table["Gram"][target] = collect([r for r in dino if r["_src"] == "exp3"], "gram_correct", target)
        table["kNN"][target] = collect([r for r in dino if r["_src"] == "exp3"], "knn_correct", target)
        table["SigLIP"][target] = collect([r for r in sig if r["_src"] == "exp3"], "correct", target)
    for target, src in (("LL", "LL"), ("LSDA-Long", "LSDA_Long")):
        sub_tex = [r for r in tex if r["_src"] == src]
        sub_dino = [r for r in dino if r["_src"] == src]
        sub_sig = [r for r in sig if r["_src"] == src]
        table["LBP_GLCM"][target] = drift([r.get("correct") for r in sub_tex])
        table["Gram"][target] = drift([r.get("gram_correct") for r in sub_dino])
        table["kNN"][target] = drift([r.get("knn_correct") for r in sub_dino])
        table["SigLIP"][target] = drift([r.get("correct") for r in sub_sig])

    # ---- VLM: SS from census; others from inject3.json
    census = rows(args.census)
    ss_flags = []
    for r in census:
        if r["id"].endswith("_SS"):
            ss_flags.append(vlm_c1(r))
    table["VLM"]["SS"] = drift(ss_flags)
    inj = json.loads(Path(args.inject3).read_text(encoding="utf-8"))["c1"]
    for target in ("LSDA-attrs", "LSDA-Long", "LL"):
        table["VLM"][target] = {"n": inj[target]["n"], "success": inj[target]["success"], "drift": inj[target]["drift"],
                                "repair_rate": inj[target]["repair_rate"], "harm_rate": inj[target]["harm_rate"]}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- markdown table
    lines = ["| metric | SS | LSDA-attrs | LSDA-Long | LL |", "|---|---:|---:|---:|---:|"]
    for m in table:
        cells = []
        for a in arms:
            c = table[m][a]
            cells.append("--" if not c or c["drift"] is None else f"{100*c['drift']:.1f} (n={c['n']})")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    (args.out.parent / "table_inject3.md").write_text(md + "\n", encoding="utf-8")
    print(md)

    # ---- grouped bar chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    metrics = ["VLM", "LBP_GLCM", "Gram", "kNN", "SigLIP"]
    labels = ["VLM (c1)", "LBP/GLCM", "Gram", "k-NN", "SigLIP"]
    colors = {"SS": "#9aa5b1", "LSDA-attrs": "#c98b48", "LSDA-Long": "#2f9e6e", "LL": "#4a6fa5"}
    x = np.arange(len(metrics))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=180)
    for j, a in enumerate(arms):
        vals = [100 * (table[m][a]["drift"] or 0) for m in metrics]
        bars = ax.bar(x + (j - 1.5) * w, vals, w, label=a, color=colors[a], edgecolor="#333333", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Drift / mis-binding rate (%)")
    ax.set_title("Knowledge injection location: 5 metrics x 4 arms (100 pairs x 3 seeds)")
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False, ncol=4)
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure)
    print("saved", args.out, args.figure)


if __name__ == "__main__":
    main()
