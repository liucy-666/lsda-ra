"""Analyze culture-axis projections: normal vs abnormal residual accumulation + attention amplification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

WINDOW = (24, 25, 26, 27)


def seed_curve(rec, key, layer, window=WINDOW):
    steps = rec.get(key, {}).get(str(layer), {})
    vals = [steps[str(t)] for t in window if str(t) in steps]
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fig-dir", type=Path, required=True)
    ap.add_argument("--cohorts", type=Path, default=None, help="deepdive_analysis.json for new-judge cohort labels")
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(38)))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    overrides = {}
    if args.cohorts and args.cohorts.exists():
        raw = json.loads(args.cohorts.read_text(encoding="utf-8"))
        for key, v in raw.items():
            pair, seed = key.split("|")
            overrides[(pair, seed)] = "failed" if v.get("leaked") else "normal"

    summary = {}
    for f in sorted(args.axis_dir.glob("axis_pair*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        pair = data["pair"]
        projs = data["projections"]
        failed, normal = {}, {}
        for s, r in projs.items():
            cohort = overrides.get((pair, s), r.get("cohort"))
            (failed if cohort == "failed" else normal)[s] = r

        def curve(seeds, key):
            out = {}
            for layer in args.layers:
                vals = [seed_curve(r, key, layer) for r in seeds.values()]
                vals = [v for v in vals if v is not None]
                out[layer] = (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, len(vals))
            return out

        p_in_f, p_in_n = curve(failed, "p_in"), curve(normal, "p_in")
        p_attn_f = curve(failed, "p_attn")
        p_ffn_f = curve(failed, "p_ffn")

        # amplification: correlation between p_attn(L) and p_in(L-1) across failed seeds
        corrs = {}
        for layer in args.layers:
            if layer == 0:
                continue
            xs, ys = [], []
            for r in failed.values():
                a = seed_curve(r, "p_attn", layer)
                b = seed_curve(r, "p_in", layer - 1)
                if a is not None and b is not None:
                    xs.append(b)
                    ys.append(a)
            if len(xs) > 2 and np.std(xs) > 0 and np.std(ys) > 0:
                corrs[layer] = float(np.corrcoef(xs, ys)[0, 1])

        summary[pair] = {
            "n_failed": len(failed), "n_normal": len(normal),
            "p_in_failed": {str(k): v for k, v in p_in_f.items()},
            "p_in_normal": {str(k): v for k, v in p_in_n.items()},
            "p_attn_failed": {str(k): v for k, v in p_attn_f.items()},
            "p_ffn_failed": {str(k): v for k, v in p_ffn_f.items()},
            "attn_resid_corr": {str(k): v for k, v in corrs.items()},
        }

        # ---- figure: p_in vs layer (failed vs normal)
        fig, ax = plt.subplots(figsize=(6.6, 4.0))
        layers = args.layers
        mf = np.array([p_in_f[l][0] for l in layers])
        sf = np.array([p_in_f[l][1] for l in layers])
        mn = np.array([p_in_n[l][0] for l in layers])
        sn = np.array([p_in_n[l][1] for l in layers])
        ax.plot(layers, mf, color="#D55E00", lw=2.0, label=f"abnormal (n={len(failed)})")
        ax.fill_between(layers, mf - sf, mf + sf, color="#D55E00", alpha=0.18)
        ax.plot(layers, mn, color="#0072B2", lw=2.0, label=f"normal (n={len(normal)})")
        ax.fill_between(layers, mn - sn, mn + sn, color="#0072B2", alpha=0.18)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Residual-stream projection  $p_{in}$  (→ A)")
        ax.set_title(f"{pair}: residual culture-axis projection across layers\n(right-object region, mean over steps {list(WINDOW)})")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(args.fig_dir / f"axis_{pair}.png", dpi=200)
        plt.close(fig)

        print(json.dumps({"pair": pair, "n_failed": len(failed), "n_normal": len(normal),
                          "p_in_failed_last": p_in_f[layers[-1]][0], "p_in_normal_last": p_in_n[layers[-1]][0]}))

    (args.out_dir / "axis_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("saved", args.out_dir / "axis_summary.json")


if __name__ == "__main__":
    main()
