"""Final data figures for the deep-dive mechanism analysis.

Fig 1: internal probe route-vs-value share across 5 cultural pairs (core evidence)
Fig 2: new-judge leakage + inclusion per pair
Fig 3: corrected h_fix outcomes (honest: no repair / corruption)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXP = Path(r"D:\Python\MMDIT\experiment\2026_9_10_EXP_1_CAUSAL")
PROBE = EXP / "deepdive" / "probe"
ANALYSIS = EXP / "deepdive" / "deepdive_analysis.json"
FIG = EXP / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, VERM, GREEN = "#0072B2", "#E69F00", "#D55E00", "#009E73"


def fig1_probe():
    pairs, route, value = [], [], []
    for f in sorted(PROBE.glob("probe_pair*_summary.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        rs = [v["route_share"] for v in data.values() if v.get("route_share") is not None]
        vs = [v["value_share"] for v in data.values() if v.get("value_share") is not None]
        pairs.append(f.stem.replace("probe_", "").replace("_summary", ""))
        route.append(np.mean(rs))
        value.append(np.mean(vs))

    route, value = np.array(route), np.array(value)
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    xs = np.arange(len(pairs))
    w = 0.6
    ax.bar(xs, route, w, color=BLUE, edgecolor="black", linewidth=0.6, label="routing (W)", zorder=3)
    ax.bar(xs, value, w, bottom=route, color=ORANGE, edgecolor="black", linewidth=0.6, label="content (V)", zorder=3)
    for x, r, v in zip(xs, route, value):
        ax.text(x, r / 2, f"{r*100:.0f}", ha="center", va="center", fontsize=8, color="white", zorder=4)
        ax.text(x, r + v / 2, f"{v*100:.0f}", ha="center", va="center", fontsize=8, color="white", zorder=4)
    ax.axhline(0.5, color="#888888", lw=0.9, ls="--", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([p.replace("pair_", "pair ") for p in pairs], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of written-output change")
    ax.set_title("Content (Value) vs routing (Attention) contribution to the attention write\n"
                 f"(5 cultural pairs, N={len(pairs)} pairs; mean value {value.mean()*100:.1f}% vs route {route.mean()*100:.1f}%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_deepdive_probe_5pairs.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig1 saved", {"route": float(route.mean()), "value": float(value.mean())})


def fig2_leakage():
    raw = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    by = {}
    for key, v in raw.items():
        p = key.split("|")[0]
        e = by.setdefault(p, {"n": 0, "leak": 0, "incl": 0})
        e["n"] += 1
        e["leak"] += int(v["leaked"])
        e["incl"] += int(v["included"])
    pairs = sorted(by)
    leak = np.array([by[p]["leak"] for p in pairs])
    ok = np.array([by[p]["n"] - by[p]["leak"] for p in pairs])
    incl = np.array([by[p]["incl"] for p in pairs])

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    xs = np.arange(len(pairs))
    w = 0.6
    ax.bar(xs, leak, w, color=VERM, edgecolor="black", linewidth=0.6, label="leaked (dual-judge)", zorder=3)
    ax.bar(xs, ok, w, bottom=leak, color="#CCCCCC", edgecolor="black", linewidth=0.6, label="not leaked", zorder=3)
    for x, l, i in zip(xs, leak, incl):
        ax.text(x, l / 2, str(l), ha="center", va="center", fontsize=9, color="white", zorder=4)
        ax.text(x, l + 0.25, f"incl {i}", ha="center", va="bottom", fontsize=7.5, color="#333333")
    ax.set_xticks(xs)
    ax.set_xticklabels([p.replace("pair_", "pair ") for p in pairs], fontsize=9)
    ax.set_ylabel("Seeds (of 9 per pair)")
    ax.set_ylim(0, 10)
    ax.set_title("Regenerated 1024 samples: dual-judge leakage under the new judges\n"
                 "(GPT-5.4 + Gemini-3.5-flash; leaked = both raters fail the same side)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_deepdive_leakage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig2 saved", {"total_leaked": int(leak.sum()), "total_included": int(incl.sum())})


def fig3_hfix():
    rows = [
        ("Last layer only (L37 / t27)", "still blue-and-white — NO repair", VERM),
        ("Late window (L33-36 / t24-27)", "corrupted (color noise)", "#999999"),
        ("All 38 layers", "corrupted (color noise)", "#999999"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ys = np.arange(len(rows))[::-1]
    for y, (cond, outcome, color) in zip(ys, rows):
        ax.barh(y, 1.0, height=0.5, color=color, edgecolor="black", linewidth=0.7, zorder=3)
        ax.text(0.02, y, cond, ha="left", va="center", fontsize=9, color="white", zorder=4)
        ax.text(1.06, y, outcome, ha="left", va="center", fontsize=9, color="#222222", zorder=4)
    ax.set_xlim(0, 2.4)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Corrected h_fix (residual-stream replacement): no repair\n"
                 "(earlier 'repair' was a dimension bug — retracted)", fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_deepdive_hfix_corrected.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig3 saved")


if __name__ == "__main__":
    fig1_probe()
    fig2_leakage()
    fig3_hfix()
