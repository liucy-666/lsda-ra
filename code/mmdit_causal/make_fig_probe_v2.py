"""Direction-aware probe figures: how much each swap closes the gap toward the donor."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXP = Path(r"D:\Python\MMDIT\experiment\2026_9_10_EXP_1_CAUSAL")
PROBE = EXP / "deepdive" / "probe_v2"
FIG = EXP / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE = "#0072B2", "#E69F00"


def main():
    pairs, pw, pv = [], [], []
    all_w, all_v = [], []
    per_seed = []
    for f in sorted(PROBE.glob("probe_pair*_summary.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        w = [v["mean_proj_W"] for v in data.values()]
        v = [v["mean_proj_V"] for v in data.values()]
        pairs.append(f.stem.replace("probe_", "").replace("_summary", "").replace("pair", "pair "))
        pw.append(np.mean(w))
        pv.append(np.mean(v))
        all_w += w
        all_v += v
        for s, rec in data.items():
            per_seed.append((rec["mean_proj_W"], rec["mean_proj_V"]))

    pw, pv = np.array(pw), np.array(pv)
    all_w, all_v = np.array(all_w), np.array(all_v)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.6, 4.2), gridspec_kw={"width_ratios": [1.25, 1.0]})

    xs = np.arange(len(pairs))
    w = 0.36
    axA.bar(xs - w / 2, pw, w, color=BLUE, edgecolor="black", linewidth=0.6, label="routing (W) swap", zorder=3)
    axA.bar(xs + w / 2, pv, w, color=ORANGE, edgecolor="black", linewidth=0.6, label="content (V) swap", zorder=3)
    axA.axhline(0, color="black", lw=0.9)
    axA.axhline(1, color="#888888", lw=0.8, ls="--")
    axA.set_xticks(xs)
    axA.set_xticklabels(pairs, fontsize=9)
    axA.set_ylabel("Fraction of gap closed toward donor")
    axA.set_title("(A) Direction-aware: how much each swap moves\nthe output toward the correct (donor) representation")
    axA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=9)

    axB.scatter(all_w, all_v, s=34, color="#333333", zorder=3)
    lim = max(0.5, float(max(np.abs(all_w).max(), np.abs(all_v).max())) * 1.15)
    axB.plot([-lim, lim], [-lim, lim], ls="--", color="#999999", lw=1, label="equal")
    axB.axhline(0, color="#cccccc", lw=0.8)
    axB.axvline(0, color="#cccccc", lw=0.8)
    axB.set_xlim(-lim, lim)
    axB.set_ylim(-lim, lim)
    axB.set_xlabel("routing swap (proj W)")
    axB.set_ylabel("content swap (proj V)")
    axB.set_title(f"(B) Per-seed (N={len(all_w)})\nmean: W={all_w.mean():.2f}, V={all_v.mean():.2f}")
    axB.legend(fontsize=8, loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(FIG / "fig_probe_direction_aware.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"proj_W": float(all_w.mean()), "proj_V": float(all_v.mean()),
                      "std_W": float(all_w.std(ddof=1)), "std_V": float(all_v.std(ddof=1)),
                      "ratio": float(all_v.mean() / all_w.mean()), "n": len(all_w)}))


if __name__ == "__main__":
    main()
