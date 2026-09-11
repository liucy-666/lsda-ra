"""Analyze E2 2x2 results: per-arm recovery, factorial effects, sign test, figures."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ARMS = ["baseline", "w_fix", "v_fix", "both_fix"]
ARM_LABEL = {"baseline": "baseline", "w_fix": "W-fix\n(route)", "v_fix": "V-fix\n(content)", "both_fix": "Both-fix"}
COLORS = {"baseline": "#9e9e9e", "w_fix": "#4c72b0", "v_fix": "#dd8452", "both_fix": "#55a868"}


def binding_of(rater_row) -> float | None:
    """right_is_b, or 1 - right_is_a if the rater returned the swapped key."""
    if not isinstance(rater_row, dict):
        return None
    v = rater_row.get("right_is_b")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    a = rater_row.get("right_is_a")
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return 1.0 - float(a)
    return None


def parse_scores(path: Path) -> dict:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        m = re.match(r"e2_p(\d+)_s(\d+)_(.+)$", r["id"])
        if not m:
            continue
        pair, seed, arm = int(m.group(1)), int(m.group(2)), m.group(3)
        vals, raters = [], {}
        for key, val in r.items():
            if key in ("id", "image"):
                continue
            v = binding_of(val)
            if v is not None:
                vals.append(v)
                raters[key] = v
        if vals:
            rows.setdefault(pair, {}).setdefault(seed, {})[arm] = {
                "mean": float(np.mean(vals)),
                "n_raters": len(vals),
                **raters,
            }
    return rows


def sign_test(k: int, n: int) -> float:
    if n == 0:
        return float("nan")
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fig-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    data = parse_scores(args.scores)
    summary = {"pairs": {}}

    for pair, seeds in sorted(data.items()):
        complete = {s: v for s, v in seeds.items() if all(a in v for a in ARMS)}
        if not complete:
            continue
        seed_list = sorted(complete)
        per_arm = {a: np.array([complete[s][a]["mean"] for s in seed_list]) for a in ARMS}
        rec = {a: per_arm[a] - per_arm["baseline"] for a in ARMS}
        r_w, r_v, r_both = rec["w_fix"], rec["v_fix"], rec["both_fix"]
        interaction = r_both - r_w - r_v
        # flips: content/w-fix counts where recovery > 0
        flips = {a: int((rec[a] > 1e-6).sum()) for a in ARMS}
        p_vals = {a: sign_test(flips[a], len(seed_list)) for a in ARMS}

        # route vs value relative contribution
        denom = float(np.abs(r_w).mean() + np.abs(r_v).mean())
        route_pct = float(np.abs(r_w).mean() / denom) if denom > 0 else float("nan")
        value_pct = float(np.abs(r_v).mean() / denom) if denom > 0 else float("nan")

        summary["pairs"][pair] = {
            "n_seeds": len(seed_list),
            "seeds": seed_list,
            "per_arm_mean": {a: float(per_arm[a].mean()) for a in ARMS},
            "recovery_mean": {a: float(rec[a].mean()) for a in ARMS},
            "recovery_ci95": {a: float(1.96 * rec[a].std(ddof=1) / math.sqrt(len(seed_list))) if len(seed_list) > 1 else 0.0 for a in ARMS},
            "flips": flips,
            "sign_p": p_vals,
            "interaction_mean": float(interaction.mean()),
            "route_pct": route_pct,
            "value_pct": value_pct,
            "per_seed": {
                str(s): {a: complete[s][a] for a in ARMS} for s in seed_list
            },
        }

        # ---- Fig 2: per-arm recovery bars
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        xs = np.arange(len(ARMS))
        means = [rec[a].mean() for a in ARMS]
        errs = [summary["pairs"][pair]["recovery_ci95"][a] for a in ARMS]
        ax.bar(xs, means, width=0.6, color=[COLORS[a] for a in ARMS], edgecolor="black", linewidth=0.7)
        ax.errorbar(xs, means, yerr=errs, fmt="none", ecolor="#444444", elinewidth=1.0, capsize=3, capthick=1.0)
        ax.axhline(0, color="black", lw=0.9)
        ax.set_xticks(xs)
        ax.set_xticklabels([ARM_LABEL[a] for a in ARMS])
        ax.set_ylabel("Binding recovery Δ (vs baseline)")
        ax.set_title(f"Pair {pair}: 2×2 causal recovery (N={len(seed_list)})")
        ymax = max(m + e for m, e in zip(means, errs))
        ymin = min(0.0, min(m - e for m, e in zip(means, errs)))
        span = ymax - ymin
        ax.set_ylim(ymin - 0.05 * span, ymax + 0.30 * span)
        for x, m, e, a in zip(xs, means, errs, ARMS):
            ax.text(x, m + e + 0.03 * span, f"{m:+.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax.text(x, m + e + 0.14 * span, f"p={p_vals[a]:.3f}", ha="center", va="bottom", fontsize=7, color="#555555")
        fig.tight_layout()
        fig.savefig(args.fig_dir / f"fig2_pair{pair}_recovery.png", dpi=200)
        plt.close(fig)

        # ---- Fig 3: per-seed scatter w_fix vs v_fix
        fig, ax = plt.subplots(figsize=(4.8, 4.6))
        ax.scatter(r_w, r_v, s=46, color="#333333", zorder=3)
        lim = max(0.2, float(max(np.abs(r_w).max(), np.abs(r_v).max())) * 1.18)
        ax.plot([-lim, lim], [-lim, lim], ls="--", color="#999999", lw=1, label="equal effect")
        ax.axhline(0, color="#cccccc", lw=0.8)
        ax.axvline(0, color="#cccccc", lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("W-fix recovery Δ (routing)")
        ax.set_ylabel("V-fix recovery Δ (content)")
        ax.set_title(f"Pair {pair}: per-seed recovery")
        ax.legend(fontsize=8, loc="lower right", frameon=False)
        fig.tight_layout()
        fig.savefig(args.fig_dir / f"fig3_pair{pair}_scatter.png", dpi=200)
        plt.close(fig)

        # ---- Fig 4: responsibility stacked bar
        fig, ax = plt.subplots(figsize=(2.8, 3.8))
        ax.bar([0], [route_pct], color=COLORS["w_fix"], edgecolor="black", linewidth=0.7, label="routing (W)")
        ax.bar([0], [value_pct], bottom=[route_pct], color=COLORS["v_fix"], edgecolor="black", linewidth=0.7, label="content (V)")
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_ylabel("Relative contribution")
        ax.set_title(f"Pair {pair}: route vs content")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.06), frameon=False)
        ax.text(0, route_pct / 2, f"{route_pct*100:.0f}%", ha="center", va="center", fontsize=11, color="white", fontweight="bold")
        ax.text(0, route_pct + value_pct / 2, f"{value_pct*100:.0f}%", ha="center", va="center", fontsize=11, color="white", fontweight="bold")
        fig.tight_layout()
        fig.savefig(args.fig_dir / f"fig4_pair{pair}_responsibility.png", dpi=200)
        plt.close(fig)

    (args.out_dir / "e2_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({"pairs": {str(p): {"n": v["n_seeds"], "recovery": v["recovery_mean"], "route_pct": v["route_pct"], "value_pct": v["value_pct"]} for p, v in summary["pairs"].items()}}, indent=1))


if __name__ == "__main__":
    main()
