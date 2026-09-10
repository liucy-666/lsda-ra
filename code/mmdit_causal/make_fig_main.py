"""Main result figure: (A) internal route-vs-value share, (B) final-image 2x2 recovery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--internal-summary", type=Path, required=True)
    ap.add_argument("--e2-summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    internal = json.loads(args.internal_summary.read_text(encoding="utf-8"))
    e2 = json.loads(args.e2_summary.read_text(encoding="utf-8"))["pairs"]["1"]

    seeds = sorted(internal, key=lambda s: int(s))
    route = np.array([internal[s]["route_share"] for s in seeds])
    value = np.array([internal[s]["value_share"] for s in seeds])
    xs = np.arange(len(seeds))

    arms = ["baseline", "w_fix", "v_fix", "both_fix"]
    labels = ["baseline", "W-fix\n(route)", "V-fix\n(content)", "Both-fix"]
    colors = ["#9e9e9e", "#4c72b0", "#dd8452", "#55a868"]
    means = [e2["recovery_mean"][a] for a in arms]
    errs = [e2["recovery_ci95"][a] for a in arms]
    pvals = e2["sign_p"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.2), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # Panel A
    w = 0.62
    axA.bar(xs, route, w, color="#4c72b0", edgecolor="black", linewidth=0.6, label="routing (W)", zorder=3)
    axA.bar(xs, value, w, bottom=route, color="#dd8452", edgecolor="black", linewidth=0.6, label="content (V)", zorder=3)
    for x, r, v in zip(xs, route, value):
        axA.text(x, r / 2, f"{r*100:.0f}", ha="center", va="center", fontsize=7, color="white", zorder=4)
        axA.text(x, r + v / 2, f"{v*100:.0f}", ha="center", va="center", fontsize=7, color="white", zorder=4)
    axA.axhline(0.5, color="#888888", lw=0.9, ls="--", zorder=1)
    axA.set_xticks(xs)
    axA.set_xticklabels([f"s{s[-2:]}" for s in seeds])
    axA.set_ylim(0, 1)
    axA.set_ylabel("Share of written-output change")
    axA.set_title("(A) Internal decomposition: routing vs content\n(L33-36, t24-27; N=9 seeds)")
    axA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=9)

    # Panel B
    xb = np.arange(len(arms))
    axB.bar(xb, means, width=0.6, color=colors, edgecolor="black", linewidth=0.7, zorder=3)
    axB.errorbar(xb, means, yerr=errs, fmt="none", ecolor="#444444", elinewidth=1.0, capsize=3, capthick=1.0, zorder=4)
    axB.axhline(0, color="black", lw=0.9)
    axB.set_xticks(xb)
    axB.set_xticklabels(labels)
    axB.set_ylabel("Binding recovery Δ (vs baseline)")
    axB.set_title("(B) Final-image 2×2 recovery\n(single point L36/t26; N=9)")
    ymax = max(m + e for m, e in zip(means, errs))
    ymin = min(0.0, min(m - e for m, e in zip(means, errs)))
    span = ymax - ymin
    axB.set_ylim(ymin - 0.05 * span, ymax + 0.32 * span)
    for x, m, e, a in zip(xb, means, errs, arms):
        axB.text(x, m + e + 0.04 * span, f"{m:+.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        axB.text(x, m + e + 0.15 * span, f"p={pvals[a]:.3f}", ha="center", va="bottom", fontsize=7, color="#555555")

    fig.suptitle("Cultural binding drift: content (Value) dominates routing (Attention)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"out": str(args.out), "value_share": float(value.mean()), "route_share": float(route.mean())}))


if __name__ == "__main__":
    main()
