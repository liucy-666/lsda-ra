"""Per-metric bar charts (4 arms) for EXP_6 from analysis/metrics_all_arms.json."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {"VLM": "VLM (c1)", "LBP_GLCM": "LBP/GLCM", "Gram": "Gram", "kNN": "k-NN", "SigLIP": "SigLIP MaSC-CP"}
ARMS = ["SS", "LSDA-attrs", "LSDA-Long", "LL"]
COLORS = {"SS": "#9aa5b1", "LSDA-attrs": "#c98b48", "LSDA-Long": "#2f9e6e", "LL": "#4a6fa5"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    table = json.loads(args.table.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for m, per in table.items():
        vals = [100 * (per[a]["drift"] or 0) for a in ARMS]
        ns = [per[a]["n"] for a in ARMS]
        fig, ax = plt.subplots(figsize=(5.6, 4.0), dpi=180)
        bars = ax.bar(ARMS, vals, color=[COLORS[a] for a in ARMS], edgecolor="#333333", linewidth=0.5)
        for b, v, n in zip(bars, vals, ns):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}\n(n={n})", ha="center", fontsize=8)
        ax.set_ylabel("Drift (%)")
        ax.set_title(f"{LABELS.get(m, m)} — knowledge injection location")
        ax.set_ylim(0, max(vals) * 1.28)
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(fontsize=9)
        fig.tight_layout()
        out = args.out_dir / f"fig_metric_{m.lower()}.png"
        fig.savefig(out)
        plt.close(fig)
        print("saved", out)


if __name__ == "__main__":
    main()
