"""Per-class (KA/ME) drift figure from analysis/metrics.json."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, r"D:\Python\MMDIT\code\metrics")
from palette import METHOD_COLORS, EDGE, style_axes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    m = json.loads(args.metrics.read_text(encoding="utf-8"))
    per = m["per_class"]
    classes = [c for c in ("ME", "KA") if c in per]
    methods = [("native_drift", "Native SS"), ("uniform_drift", "LSDA uniform"), ("routed_drift", "Ours (routed)")]
    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=140)
    method_colors = ["#9E9E9E", "#56B4E9", "#0072B2"]
    for j, (key, label) in enumerate(methods):
        vals = [per[c][key] * 100 for c in classes]
        bars = ax.bar(x + (j - 1) * width, vals, width, label=label, color=method_colors[j], edgecolor=EDGE, linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", fontsize=9, color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c} (n={per[c]['n']})" for c in classes])
    ax.set_ylabel("Drift rate (%)")
    ax.set_title("Per-class drift: ME vs KA")
    ax.set_ylim(0, 115)
    style_axes(ax)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
