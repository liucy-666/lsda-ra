"""Aggregate the 5 metrics x 3 arms drift table and draw the comparison figure.

Metrics: VLM (dual-rater), LBP/GLCM, Gram, kNN, SigLIP MaSC-CP.
Arms: SS (native), LSDA (unified knowledge injection), Binding (attention binding).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    if path and Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def vlm_binding(row):
    """Project-standard strict intersection: a rater fails if either side is wrong;
    the sample fails only when BOTH raters fail; correct = not sample failure."""
    if not row:
        return None
    rater_fail = []
    for r in ("gpt54", "gemini35"):
        d = row.get(r) or {}
        if not isinstance(d, dict):
            return None
        rv = d.get("right_is_b")
        lv = d.get("left_is_a")
        if not isinstance(rv, (int, float)) and isinstance(lv, (int, float)):
            rv = 1.0 - lv
        if not isinstance(rv, (int, float)) or not isinstance(lv, (int, float)):
            return None
        rater_fail.append(float(rv) < 0.5 or float(lv) < 0.5)
    return not all(rater_fail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", type=Path, required=True)
    ap.add_argument("--census-scores", type=Path, required=True)
    ap.add_argument("--repair-scores", type=Path, nargs="+", required=True)
    ap.add_argument("--binding-scores", type=Path, default=None)
    ap.add_argument("--texture", type=Path, required=True)
    ap.add_argument("--dino", type=Path, required=True)
    ap.add_argument("--siglip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figure", type=Path, required=True)
    args = ap.parse_args()

    cls = json.loads(args.classification.read_text(encoding="utf-8"))
    census = {r["id"]: r for r in load_jsonl(args.census_scores)}
    rep = {}
    for p in args.repair_scores:
        rep.update({r["id"]: r for r in load_jsonl(p)})
    bind = {r["id"]: r for r in load_jsonl(args.binding_scores)} if args.binding_scores else {}

    tex = {r["id"]: r for r in load_jsonl(args.texture)}
    dino = {r["id"]: r for r in load_jsonl(args.dino)}
    sig = {r["id"]: r for r in load_jsonl(args.siglip)}

    arms = ("SS", "LSDA", "Binding")
    acc = {m: {a: [0, 0] for a in arms} for m in ("VLM", "LBP_GLCM", "Gram", "kNN", "SigLIP")}

    def add(metric, arm, correct):
        if correct is None:
            return
        acc[metric][arm][1] += 1
        acc[metric][arm][0] += 1 if correct else 0

    for rec in cls.values():
        label = rec.get("label")
        if label not in ("KA", "ME", "BC"):
            continue
        pair, seed = int(rec["pair"]), int(rec["seed"])
        rid = f"p{pair:03d}_s{seed}"
        ss_row = census.get(f"cen_p{pair:03d}_s{seed}_SS")
        # SS
        add("VLM", "SS", vlm_binding(ss_row))
        add("LBP_GLCM", "SS", (tex.get(f"SS_{rid}") or {}).get("correct"))
        add("Gram", "SS", (dino.get(f"SS_{rid}") or {}).get("gram_correct"))
        add("kNN", "SS", (dino.get(f"SS_{rid}") or {}).get("knn_correct"))
        add("SigLIP", "SS", (sig.get(f"SS_{rid}") or {}).get("correct"))
        # LSDA (unified knowledge injection)
        if label == "ME":
            run = rep.get(f"lsda_meattrs_p{pair:03d}_s{seed}")
        elif label == "KA":
            run = rep.get(f"lsda_ka_p{pair:03d}_s{seed}")
        else:
            run = ss_row
        add("VLM", "LSDA", vlm_binding(run) if run else None)
        add("LBP_GLCM", "LSDA", (tex.get(f"LSDA_{rid}") or {}).get("correct"))
        add("Gram", "LSDA", (dino.get(f"LSDA_{rid}") or {}).get("gram_correct"))
        add("kNN", "LSDA", (dino.get(f"LSDA_{rid}") or {}).get("knn_correct"))
        add("SigLIP", "LSDA", (sig.get(f"LSDA_{rid}") or {}).get("correct"))
        # Binding
        b = bind.get(f"bind_p{pair:03d}_s{seed}")
        add("VLM", "Binding", vlm_binding(b) if b else None)
        add("LBP_GLCM", "Binding", (tex.get(f"Binding_{rid}") or {}).get("correct"))
        add("Gram", "Binding", (dino.get(f"Binding_{rid}") or {}).get("gram_correct"))
        add("kNN", "Binding", (dino.get(f"Binding_{rid}") or {}).get("knn_correct"))
        add("SigLIP", "Binding", (sig.get(f"Binding_{rid}") or {}).get("correct"))

    table = {}
    for m, per_arm in acc.items():
        table[m] = {}
        for a in arms:
            ok, n = per_arm[a]
            table[m][a] = {"n": n, "success": ok, "drift": (1 - ok / n) if n else None}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(table, ensure_ascii=False, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from palette import ARM_COLORS, EDGE, style_axes

    metric_labels = {"VLM": "VLM (dual)", "LBP_GLCM": "LBP/GLCM", "Gram": "Gram", "kNN": "k-NN (proto)", "SigLIP": "SigLIP CP"}
    metrics = list(metric_labels)
    x = np.arange(len(metrics))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=200)
    for j, a in enumerate(arms):
        vals = [(table[m][a]["drift"] or 0) * 100 for m in metrics]
        bars = ax.bar(x + (j - 1) * width, vals, width, label=a, color=ARM_COLORS[a], edgecolor=EDGE, linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", fontsize=7.5, color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[m] for m in metrics])
    ax.set_ylabel("Drift / mis-binding rate (%)")
    ax.set_title("5 metrics x 3 arms (n=300, 100 pairs x 3 seeds)")
    ax.set_ylim(0, 105)
    style_axes(ax)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, bbox_inches="tight")
    print(f"saved {args.figure}")


if __name__ == "__main__":
    main()
