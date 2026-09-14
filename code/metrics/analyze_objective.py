"""Analyze objective binding metrics (CLIP/DINO) and their agreement with the VLM judge.

Inputs: classification, VLM census scores, VLM repair scores, objective scores (merged).
Outputs: objective drift per method, VLM-vs-objective agreement, and a comparison figure.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def vlm_binding(row):
    vals = []
    for r in ("gpt54", "gemini35"):
        d = row.get(r) or {}
        left, right = d.get("left_is_a"), d.get("right_is_b")
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return None
        vals.append(left >= 0.5 and right >= 0.5)
    return all(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", type=Path, required=True)
    ap.add_argument("--census-scores", type=Path, required=True)
    ap.add_argument("--repair-scores", type=Path, required=True)
    ap.add_argument("--objective", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figure", type=Path, required=True)
    args = ap.parse_args()

    cls = json.loads(args.classification.read_text(encoding="utf-8"))
    census = load_jsonl(args.census_scores)
    rep = load_jsonl(args.repair_scores)
    obj = {}
    for p in args.objective:
        for key, row in load_jsonl(p).items():
            obj.setdefault(key, {}).update(row)

    rows = []
    for rec in sorted(cls.values(), key=lambda r: (int(r["pair"]), int(r["seed"]))):
        pair, seed, label = int(rec["pair"]), int(rec["seed"]), rec["label"]
        entries = {}
        ss = census.get(f"cen_p{pair:03d}_s{seed}_SS")
        entries["native"] = {"vlm": (not rec["leaked"]), "obj": obj.get(f"native_p{pair:03d}_s{seed}")}
        if label == "BC":
            entries["routed"] = entries["native"]
            entries["uniform"] = entries["native"]
        elif label == "ME":
            run = rep.get(f"lsda_me_p{pair:03d}_s{seed}")
            o = obj.get(f"routed_p{pair:03d}_s{seed}")
            entries["routed"] = {"vlm": vlm_binding(run) if run else None, "obj": o}
            entries["uniform"] = entries["routed"]
        elif label == "KA":
            rk = rep.get(f"lsda_ka_p{pair:03d}_s{seed}")
            ru = rep.get(f"lsda_uniform_p{pair:03d}_s{seed}")
            entries["routed"] = {"vlm": vlm_binding(rk) if rk else None, "obj": obj.get(f"routed_p{pair:03d}_s{seed}")}
            entries["uniform"] = {"vlm": vlm_binding(ru) if ru else None, "obj": obj.get(f"uniform_p{pair:03d}_s{seed}")}
        else:
            continue
        for method, e in entries.items():
            o = e["obj"] or {}
            rows.append({
                "method": method, "pair": pair, "seed": seed, "label": label,
                "vlm_ok": e["vlm"],
                "obj_clip_i": (o.get("attr_clip_i") == "correct") if o else None,
                "obj_clip_t": (o.get("attr_clip_t") == "correct") if o else None,
                "obj_dino": (o.get("attr_dino") == "correct") if o else None,
            })

    summary = {"n_rows": len(rows), "methods": {}}
    for method in ("native", "uniform", "routed"):
        sub = [r for r in rows if r["method"] == method]
        out = {"n": len(sub)}
        for key, name in (("vlm_ok", "vlm"), ("obj_clip_i", "clip_i"), ("obj_clip_t", "clip_t"), ("obj_dino", "dino")):
            vals = [r[key] for r in sub if r[key] is not None]
            out[name] = {"n": len(vals), "success": int(sum(vals)), "drift_rate": 1 - sum(vals) / len(vals) if vals else None}
        agree = {}
        for key, name in (("obj_clip_i", "clip_i"), ("obj_clip_t", "clip_t"), ("obj_dino", "dino")):
            pairs = [(r["vlm_ok"], r[key]) for r in sub if r[key] is not None and r["vlm_ok"] is not None]
            agree[name] = (sum(1 for a, b in pairs if a == b) / len(pairs)) if pairs else None
        out["agreement_with_vlm"] = agree
        summary["methods"][method] = out

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from palette import EDGE, SERIES_COLORS, style_axes

    methods = ["native", "uniform", "routed"]
    x = np.arange(len(methods))
    width = 0.2
    series = [("vlm", "VLM (GPT+Gemini)"), ("clip_i", "CLIP-I"), ("clip_t", "CLIP-T"), ("dino", "DINOv2")]
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=200)
    for j, (key, label) in enumerate(series):
        vals = [(summary["methods"][m].get(key, {}) or {}).get("drift_rate") for m in methods]
        vals = [0 if v is None else v * 100 for v in vals]
        ax.bar(x + (j - 1.5) * width, vals, width, label=label, color=SERIES_COLORS[j], edgecolor=EDGE, linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["Native SS", "LSDA uniform", "Ours (routed)"])
    ax.set_ylabel("Drift rate (%)")
    ax.set_title("VLM vs objective binding metrics (n=300)")
    style_axes(ax)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, bbox_inches="tight")
    print(f"saved {args.figure}")


if __name__ == "__main__":
    main()
