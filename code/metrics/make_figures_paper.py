"""Paper-level figures for Method / Experiments.

Produces:
  figures/fig_method_pipeline.png    - unified knowledge injection + local re-diffusion pipeline
  figures/fig_census_distribution.png - KA / ME / BC sample distribution
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
from palette import ARM_COLORS, EDGE, style_axes  # noqa: E402

EXP = Path(r"D:\Python\MMDIT\experiment\2026_9_12_EXP_3_KA_ME")
FIG = EXP / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec="#333333", fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.1, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="#111111", weight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p0, p1, color="#333333", style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                                 linewidth=1.1, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


def method_pipeline():
    fig, ax = plt.subplots(figsize=(12.4, 4.2), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    label = dict(fc="#F5F5F5")
    ours = dict(fc="#E3F2FD", ec=ARM_COLORS["LSDA"])

    # Row 1
    box(ax, 0.015, 0.66, 0.155, 0.22, "Scene prompt\n(A left, B right)", **label)
    box(ax, 0.215, 0.66, 0.150, 0.22, "SD3.5-large\nnative SS", **label)
    box(ax, 0.410, 0.66, 0.170, 0.22, "Binding failure?\n(strict dual VLM)", **label)
    box(ax, 0.690, 0.74, 0.200, 0.16, "keep native SS\n(no failure)", fc="#FFFFFF")

    # Row 2 (repair branch)
    box(ax, 0.330, 0.30, 0.250, 0.24,
        "Unified knowledge injection\nentity prompt =\nshort phrase + attributes", **ours)
    box(ax, 0.660, 0.30, 0.280, 0.24,
        "Local re-diffusion\nSAM-bbox region experts\n(step 0..T) + native background", **ours)
    box(ax, 0.660, 0.04, 0.280, 0.15, "Repaired composition", fc="#FFFFFF")

    arrow(ax, (0.170, 0.77), (0.215, 0.77))
    arrow(ax, (0.365, 0.77), (0.410, 0.77))
    arrow(ax, (0.580, 0.77), (0.690, 0.82), rad=-0.12)   # no
    arrow(ax, (0.495, 0.66), (0.455, 0.54), rad=0.12)     # yes
    arrow(ax, (0.580, 0.42), (0.660, 0.42))
    arrow(ax, (0.800, 0.30), (0.800, 0.19))

    ax.text(0.640, 0.865, "no", fontsize=9.5, color="#555555", bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    ax.text(0.505, 0.585, "yes", fontsize=9.5, color="#555555", bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    ax.text(0.015, 0.94, "Training-free", fontsize=9.5, color="#555555", style="italic")

    fig.tight_layout()
    out = FIG / "fig_method_pipeline.png"
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


def census_distribution():
    cls = json.loads((EXP / "analysis" / "classification.json").read_text(encoding="utf-8"))
    counts = {"BC": 0, "ME": 0, "KA": 0}
    for r in cls.values():
        if r.get("label") in counts:
            counts[r["label"]] += 1
    n = sum(counts.values())
    order = ["BC", "ME", "KA"]
    labels = ["Both correct (BC)", "Mechanism error (ME)", "Knowledge absence (KA)"]
    colors = ["#BDBDBD", "#1E88E5", "#FB8C00"]
    vals = [counts[k] for k in order]

    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=200)
    bars = ax.bar(labels, vals, color=colors, edgecolor=EDGE, linewidth=0.7, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + n * 0.012, f"{v}  ({v/n*100:.1f}%)",
                ha="center", fontsize=10, color="#222222")
    ax.set_ylabel("Number of samples")
    ax.set_title(f"Failure population (n={n}, 100 pairs x 3 seeds)")
    ax.set_ylim(0, max(vals) * 1.25)
    style_axes(ax)
    plt.xticks(fontsize=9)
    fig.tight_layout()
    out = FIG / "fig_census_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    method_pipeline()
    census_distribution()
