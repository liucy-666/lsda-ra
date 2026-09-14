"""Compute KA/ME census metrics for the 5-bar comparison and draw the main figure.

Methods: Native SS / Attention Binding (routing) / LSDA uniform / Knowledge-uniform / Ours (routed).
M1 = drift / mis-binding rate (strict dual VLM); M3 = mean structure score.
"""
import argparse
import json
from pathlib import Path

import numpy as np

METHODS = [
    ("native", "Native SS", "native_ok"),
    ("binding", "Attention Binding", "binding_ok"),
    ("uniform", "LSDA uniform", "uniform_ok"),
    ("knowledge", "Knowledge-uniform", "knowledge_ok"),
    ("routed", "Ours (routed)", "routed_ok"),
]


def binding_ok(row):
    vals = []
    for r in ("gpt54", "gemini35"):
        d = row.get(r) or {}
        left, right = d.get("left_is_a"), d.get("right_is_b")
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return None
        vals.append(left >= 0.5 and right >= 0.5)
    return all(vals)


def structure_of(row):
    if not row:
        return None
    vals = [d.get("structure") for r in ("gpt54", "gemini35") if isinstance((d := row.get(r)), dict)]
    vals = [float(v) for v in vals if isinstance(v, (int, float))]
    return float(np.mean(vals)) if vals else None


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def cluster_bootstrap(rows, key, draws=10000, seed=20260912):
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair"], []).append(1.0 if r[key] else 0.0)
    pairs = list(by_pair.keys())
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(draws):
        pick = rng.choice(len(pairs), size=len(pairs), replace=True)
        vals = []
        for i in pick:
            vals.extend(by_pair[pairs[i]])
        rates.append(float(np.mean(vals)) if vals else 0.0)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return float(lo), float(hi)


def load_jsonl(path):
    out = {}
    if path and Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", type=Path, required=True)
    ap.add_argument("--census-scores", type=Path, required=True)
    ap.add_argument("--repair-scores", type=Path, nargs="+", required=True)
    ap.add_argument("--binding-scores", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figure", type=Path, required=True)
    args = ap.parse_args()

    cls = json.loads(args.classification.read_text(encoding="utf-8"))
    census = load_jsonl(args.census_scores)
    rep = {}
    for p in args.repair_scores:
        rep.update(load_jsonl(p))
    bind = load_jsonl(args.binding_scores) if args.binding_scores else {}

    rows = []
    for key, rec in sorted(cls.items()):
        if rec.get("label") not in ("KA", "ME", "BC"):
            continue
        pair, seed = int(rec["pair"]), int(rec["seed"])
        native_ok = not rec["leaked"]
        native_struct = structure_of(census.get(f"cen_p{pair:03d}_s{seed}_SS"))
        brow = bind.get(f"bind_p{pair:03d}_s{seed}")
        binding = (not rec["leaked"]) if brow is None else bool(binding_ok(brow))
        if rec["label"] == "BC":
            uniform = knowledge = routed = True
            struct_u = struct_k = struct_r = native_struct
        elif rec["label"] == "ME":
            run_me = rep.get(f"lsda_me_p{pair:03d}_s{seed}")
            run_mea = rep.get(f"lsda_meattrs_p{pair:03d}_s{seed}")
            uniform = routed = bool(binding_ok(run_me)) if run_me else False
            knowledge = bool(binding_ok(run_mea)) if run_mea else False
            struct_u = struct_r = structure_of(run_me)
            struct_k = structure_of(run_mea)
        else:  # KA
            rk = rep.get(f"lsda_ka_p{pair:03d}_s{seed}")
            ru = rep.get(f"lsda_uniform_p{pair:03d}_s{seed}")
            routed = bool(binding_ok(rk)) if rk else False
            uniform = bool(binding_ok(ru)) if ru else False
            knowledge = routed
            struct_k = struct_r = structure_of(rk)
            struct_u = structure_of(ru)
        rows.append({
            "pair": pair, "seed": seed, "label": rec["label"],
            "native_ok": native_ok, "binding_ok": binding, "uniform_ok": bool(uniform),
            "knowledge_ok": bool(knowledge), "routed_ok": bool(routed),
            "struct_native": native_struct, "struct_uniform": struct_u,
            "struct_knowledge": struct_k, "struct_routed": struct_r,
        })

    n = len(rows)
    summary = {"n": n, "counts": {l: sum(1 for r in rows if r["label"] == l) for l in ("KA", "ME", "BC")}}
    metrics = {}
    for name, _, key in METHODS:
        k = sum(1 for r in rows if r[key])
        lo_w, hi_w = wilson(k, n)
        lo_b, hi_b = cluster_bootstrap(rows, key)
        metrics[name] = {
            "success": k, "drift_rate": 1 - k / n if n else None,
            "wilson_ci95": [1 - hi_w, 1 - lo_w], "cluster_ci95": [1 - hi_b, 1 - lo_b],
        }
    summary["metrics"] = metrics
    summary["structure_mean"] = {
        name: (float(np.mean([r[k] for r in rows if isinstance(r[k], (int, float))]))
               if any(isinstance(r[k], (int, float)) for r in rows) else None)
        for name, _, k in [
            ("native", "Native", "struct_native"), ("uniform", "U", "struct_uniform"),
            ("knowledge", "K", "struct_knowledge"), ("routed", "R", "struct_routed"),
        ]
    }
    summary["per_class"] = {}
    for label in ("KA", "ME"):
        sub = [r for r in rows if r["label"] == label]
        if sub:
            summary["per_class"][label] = {"n": len(sub)}
            for name, _, key in METHODS:
                summary["per_class"][label][f"{name}_drift"] = 1 - sum(r[key] for r in sub) / len(sub)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys as _sys
    _sys.path.insert(0, r"D:\Python\MMDIT\code\metrics")
    from palette import METHOD_COLORS, EDGE, style_axes

    labels = [m[1] for m in METHODS]
    vals = [metrics[m[0]]["drift_rate"] * 100 for m in METHODS]
    errs = [[(metrics[m[0]]["drift_rate"] - metrics[m[0]]["cluster_ci95"][0]) * 100 for m in METHODS],
            [(metrics[m[0]]["cluster_ci95"][1] - metrics[m[0]]["drift_rate"]) * 100 for m in METHODS]]
    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=140)
    bars = ax.bar(labels, vals, yerr=errs, capsize=4, color=METHOD_COLORS, edgecolor=EDGE, linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center", fontsize=10, color="#222222")
    ax.set_ylabel("Drift / mis-binding rate (%)")
    ax.set_title(f"Cultural binding census (n={n}, 100 pairs x 3 seeds; pair-clustered 95% CI)")
    ax.set_ylim(0, max(vals) * 1.3 + 2)
    style_axes(ax)
    plt.xticks(rotation=12, ha="right", fontsize=8.5)
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, bbox_inches="tight")
    print(f"saved {args.figure}")


if __name__ == "__main__":
    main()
