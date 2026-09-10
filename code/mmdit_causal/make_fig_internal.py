"""Figure: internal 2x2 route-vs-value share per seed."""
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
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--fig-dir", type=Path, required=True)
    args = ap.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    seeds = sorted(data, key=lambda s: int(s))
    route = np.array([data[s]["route_share"] for s in seeds])
    value = np.array([data[s]["value_share"] for s in seeds])

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    xs = np.arange(len(seeds))
    w = 0.62
    ax.bar(xs, route, w, color="#4c72b0", edgecolor="black", linewidth=0.6, label="routing (W)", zorder=3)
    ax.bar(xs, value, w, bottom=route, color="#dd8452", edgecolor="black", linewidth=0.6, label="content (V)", zorder=3)
    for x, r, v in zip(xs, route, value):
        ax.text(x, r / 2, f"{r*100:.0f}", ha="center", va="center", fontsize=7, color="white", zorder=4)
        ax.text(x, r + v / 2, f"{v*100:.0f}", ha="center", va="center", fontsize=7, color="white", zorder=4)
    ax.axhline(0.5, color="#888888", lw=0.9, ls="--", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"s{s[-2:]}" for s in seeds])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of output change")
    ax.set_title("Routing vs content contribution to the written attention output\n(mean over L33-36, t24-27; N=9 seeds)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.fig_dir / "fig_internal_share.png", dpi=200)
    plt.close(fig)

    print(json.dumps({
        "mean_route_share": float(route.mean()),
        "mean_value_share": float(value.mean()),
        "std_route": float(route.std(ddof=1)),
        "std_value": float(value.std(ddof=1)),
        "n_seeds": len(seeds),
    }, indent=1))


if __name__ == "__main__":
    main()
