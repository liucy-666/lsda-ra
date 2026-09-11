"""Schematic (v2): MM-DiT block data flow with numbered intervention points + legend.

Colorblind-safe palette (Okabe-Ito) and no floating annotations (legend below).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BLUE = "#0072B2"
ORANGE = "#E69F00"
VERM = "#D55E00"
GREEN = "#009E73"
GREY = "#555555"


def box(ax, x, y, w, h, text, fc, ec="black", fs=10, bold=False, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.07",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, fontweight="bold" if bold else "normal")


def arrow(ax, p, q, color="black", lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=lw, color=color, zorder=1, linestyle=ls))


def marker(ax, x, y, n, color):
    ax.plot([x], [y], marker="o", ms=11, mfc="white", mec=color, mew=1.8, zorder=6)
    ax.text(x, y, str(n), ha="center", va="center", fontsize=8.5, color=color,
            fontweight="bold", zorder=7)


def main():
    fig, ax = plt.subplots(figsize=(12.0, 4.9))
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 4.9)
    ax.axis("off")

    y, h = 2.75, 0.66

    # --- main flow (border colors match the legend: h_in=vermillion, Q,K=blue, V=orange)
    box(ax, 0.25, y, 1.05, h, "h_in", "#EDF3FA", ec=VERM, lw=1.8, fs=11, bold=True)
    box(ax, 1.70, y - 0.08, 2.30, h + 0.16, "", "white", ec=BLUE, lw=1.4)
    ax.text(2.85, y + h + 0.22, "Attention", ha="center", va="bottom", fontsize=11.5,
            color=BLUE, fontweight="bold")
    ax.text(2.24, y + 0.34, "Q,K", ha="center", va="center", fontsize=10, color=BLUE, fontweight="bold")
    ax.text(3.00, y + 0.34, "V", ha="center", va="center", fontsize=10, color=ORANGE, fontweight="bold")
    ax.text(2.62, y + 0.10, "routing", ha="center", va="top", fontsize=7.5, color=BLUE)
    ax.text(3.22, y + 0.10, "content", ha="left", va="top", fontsize=7.5, color=ORANGE)

    box(ax, 4.35, y, 1.25, h, "attn_gated", "#FBE8CE", fs=10)
    box(ax, 7.05, y, 0.95, h, "h_attn", "#EDF3FA", fs=10)
    box(ax, 8.70, y - 0.08, 1.20, h + 0.16, "FFN", "white", ec=GREEN, lw=1.4)
    ax.text(9.30, y + h + 0.22, "Feed-Forward", ha="center", va="bottom", fontsize=10,
            color=GREEN, fontweight="bold")
    box(ax, 10.55, y, 1.05, h, "h_out", "#EDF3FA", fs=11, bold=True)

    arrow(ax, (1.30, y + h / 2), (1.70, y + h / 2))
    arrow(ax, (4.00, y + h / 2), (4.35, y + h / 2))
    arrow(ax, (5.60, y + h / 2), (6.45, y + h / 2))
    arrow(ax, (7.90, y + h / 2), (8.70, y + h / 2))
    arrow(ax, (9.90, y + h / 2), (10.55, y + h / 2))

    # skip connections (residual adds)
    for cx in (6.45, 9.90):
        ax.plot([cx], [y + h / 2], marker="o", ms=15, mfc="white", mec="black", mew=1.2, zorder=5)
        ax.text(cx, y + h / 2, "+", ha="center", va="center", fontsize=12, zorder=6)
    arrow(ax, (6.45, y + h), (6.45, y + h + 0.55), color=GREY, ls="--", lw=1.0)
    arrow(ax, (1.75, y + h + 0.55), (6.45, y + h + 0.55), color=GREY, ls="--", lw=1.0)
    arrow(ax, (1.75, y + h), (1.75, y + h + 0.55), color=GREY, ls="--", lw=1.0)
    ax.text(4.10, y + h + 0.68, "h_in skip", fontsize=8.5, color=GREY)
    arrow(ax, (9.90, y + h), (9.90, y + h + 0.55), color=GREY, ls="--", lw=1.0)
    arrow(ax, (7.50, y + h + 0.55), (9.90, y + h + 0.55), color=GREY, ls="--", lw=1.0)
    arrow(ax, (7.50, y + h), (7.50, y + h + 0.55), color=GREY, ls="--", lw=1.0)
    ax.text(8.70, y + h + 0.68, "h_attn skip", fontsize=8.5, color=GREY)

    # intervention markers are encoded by border color (see legend below); no floating circles.


    # --- accumulation strip
    ys = 0.78
    ax.text(0.25, ys + 0.62, "Across 38 layers the residual stream only ADDS (history is never erased):",
            fontsize=9.5, color="#333333")
    for i in range(6):
        box(ax, 0.25 + i * 0.62, ys, 0.55, 0.42, f"L{i+1}", "#EDF3FA", fs=7.5, lw=0.9)
    arrow(ax, (4.15, ys + 0.21), (4.55, ys + 0.21), lw=1.1)
    ax.text(4.85, ys + 0.21, "h_out = h_in + attn_gated + f_gated", fontsize=9.5, va="center",
            color="#333333", family="monospace")

    # --- legend (color-coded to the box borders above)
    ly = 0.14
    items = [
        (VERM, "h_fix: replace the residual content (h_in)"),
        (BLUE, "w_fix: replace the routing (Q,K)"),
        (ORANGE, "v_fix: replace the value (V)"),
    ]
    xs = [0.35, 4.55, 8.75]
    for (c, label), x in zip(items, xs):
        ax.plot([x], [ly + 0.09], marker="o", ms=13, mfc=c, mec=c, zorder=6)
        ax.text(x + 0.22, ly + 0.09, label, ha="left", va="center", fontsize=9.5, color="#222222")

    fig.tight_layout()
    out = "fig_schematic_block.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()

