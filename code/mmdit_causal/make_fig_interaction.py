"""Figure: fixing either factor alone is insufficient; both are required.

Panel A: 4-arm binding recovery (mean +- 95% CI) with sign-test p-values.
Panel B: per-seed recovery heatmap (rows = seeds, cols = arms).
Data: e2_analysis/e2_summary.json (single-point 2x2, pair 1, N=9).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXP = Path(r"D:\Python\MMDIT\experiment\2026_9_10_EXP_1_CAUSAL")
FIG = EXP / "figures"
FIG.mkdir(parents=True, exist_ok=True)
ARMS = ["baseline", "w_fix", "v_fix", "both_fix"]
LABELS = ["baseline", "W-fix\n(routing)", "V-fix\n(content)", "Both-fix"]
COLORS = ["#9e9e9e", "#0072B2", "#E69F00", "#009E73"]


def main():
    S = json.loads((EXP / "e2_analysis" / "e2_summary.json").read_text(encoding="utf-8"))
    p = S["pairs"]["1"]
    seeds = sorted(p["per_seed"], key=int)
    means = {a: np.array([p["per_seed"][s][a]["mean"] for s in seeds]) for a in ARMS}
    rec = {a: means[a] - means["baseline"] for a in ARMS}
    pv = p["sign_p"]
    flips = p["flips"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.8, 4.3), gridspec_kw={"width_ratios": [1.0, 1.2]})

    # ---- Panel A: recovery bars
    xs = np.arange(4)
    mv = np.array([rec[a].mean() for a in ARMS])
    ci = np.array([1.96 * rec[a].std(ddof=1) / np.sqrt(len(seeds)) for a in ARMS])
    axA.bar(xs, mv, yerr=ci, color=COLORS, edgecolor="black", linewidth=0.7,
            capsize=4, error_kw={"elinewidth": 1.0, "capthick": 1.0}, zorder=3)
    axA.axhline(0, color="black", lw=0.9)
    span = float(np.max(mv + ci) - np.minimum(0.0, float(np.min(mv - ci))))
    axA.set_ylim(-0.02, (mv + ci).max() + 0.55 * span)
    for x, a in zip(xs, ARMS):
        star = "*  p<0.05" if pv[a] < 0.05 else f"n.s.  p={pv[a]:.2f}"
        axA.text(x, mv[x] + ci[x] + 0.03 * span, f"{flips[a]}/9\n{star}", ha="center", va="bottom",
                 fontsize=8, fontweight="bold" if pv[a] < 0.05 else "normal",
                 color="#111111")
    axA.set_xticks(xs)
    axA.set_xticklabels(LABELS, fontsize=8.5)
    axA.set_ylabel("Binding recovery Δ (vs baseline)")
    axA.set_title("(A) Fixing either factor alone is not significant;\nonly fixing BOTH is (8/9, p=0.02)", fontsize=9.5)

    # ---- Panel B: per-seed heatmap
    mat = np.array([[rec[a][i] for a in ARMS] for i in range(len(seeds))])
    im = axB.imshow(mat, cmap="RdBu_r", vmin=-0.35, vmax=0.35, aspect="auto")
    axB.set_xticks(range(4))
    axB.set_xticklabels(["base", "W-fix", "V-fix", "Both"], fontsize=9)
    axB.set_yticks(range(len(seeds)))
    axB.set_yticklabels([f"s{s[-2:]}" for s in seeds], fontsize=8)
    axB.set_title("(B) Per-seed recovery\n(blue = recovered, red = worsened)", fontsize=9.5)
    for i in range(len(seeds)):
        for j in range(4):
            v = mat[i, j]
            axB.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.5,
                     color="white" if abs(v) > 0.22 else "#222222")
    cbar = fig.colorbar(im, ax=axB, fraction=0.045, pad=0.03)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(FIG / "fig_interaction_2x2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"saved": "fig_interaction_2x2.png", "flips": flips,
                      "p": {a: round(pv[a], 4) for a in ARMS},
                      "recovery": {a: round(float(rec[a].mean()), 3) for a in ARMS}}))


if __name__ == "__main__":
    main()
