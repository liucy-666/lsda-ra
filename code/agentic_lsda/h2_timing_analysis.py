"""H2: Does the optimal LSDA intervention timing vary across samples?

Reads the counterfactual pilot manifest + per-replay sidecars (or a scores
overlay) and produces:

  1. data-readiness report  (which replay branches have outcome scores)
  2. per-sample timing-quality curves (binding / structure / utility vs t)
  3. optimal-timing histogram per target (A / B / AB) and per cohort
  4. aggregate mean curves with confidence bands
  5. H2 verdict stats: spread of t*, best-fixed-vs-per-sample-best gap

Score contract (expected fields on each invoke sidecar record, filled by the
scoring pass):
  binding_score:    float in [0,1]  higher = better culture binding
  structure_score:  float in [0,1]  higher = better structure / layout
  leakage_score:    float >= 0      lower = better
  artifact_score:   float >= 0      lower = better
  compute_seconds:  float >= 0      cost
  status:           "generated"     (WAIT tasks are "reference_only")

An optional --scores overlay (JSONL: {"replay_id": {...scores...}}) can be
merged on top of sidecars, so the scorer does not have to edit sidecars.

Usage:
  python h2_timing_analysis.py --manifest <counterfactual_pilot.json> \
      [--sidecars-root <record_output_dir>] [--scores <overlay.jsonl>] \
      [--out-dir <plots>] [--structure-gate 0.5]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STEPS = [0, 4, 8, 12, 16, 20, 24]
TARGETS = ["A", "B", "AB"]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "tasks" not in payload or not isinstance(payload["tasks"], list):
        raise ValueError(f"manifest {path} lacks a 'tasks' list")
    return payload


def load_sidecar_roots(root: Path) -> dict[str, dict]:
    """replay_id -> record, gathered from sidecars/<sample>/<replay>.json."""
    out: dict[str, dict] = {}
    if not root.exists():
        return out
    for sidecar in root.rglob("*.json"):
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        replay_id = record.get("replay_id")
        if replay_id:
            out[replay_id] = record
    return out


def load_score_overlay(path: Path | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path is None or not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        out[record["replay_id"]] = record
    return out


def has_scores(record: dict) -> bool:
    score = record.get("binding_score")
    return isinstance(score, (int, float)) and not isinstance(score, bool)


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def build_matrix(tasks: list[dict], sidecars: dict, overlay: dict) -> dict:
    """sample -> target -> {step: {binding, structure, ...}} plus native."""
    by_sample: dict[str, dict] = {}
    for task in tasks:
        sample = task["sample_id"]
        action = task["action"]
        merged = {**task, **sidecars.get(task["replay_id"], {}), **overlay.get(task["replay_id"], {})}
        row = {
            "replay_id": task["replay_id"],
            "transition": task.get("source_transition"),
            "cohort": task.get("cohort"),
            "has_scores": has_scores(merged),
            "binding": merged.get("binding_score"),
            "structure": merged.get("structure_score"),
            "leakage": merged.get("leakage_score"),
            "artifact": merged.get("artifact_score"),
            "compute": merged.get("compute_seconds"),
            "status": merged.get("status"),
        }
        sample_row = by_sample.setdefault(
            sample, {"native": {}, "branches": {target: {} for target in TARGETS}}
        )
        if action["kind"] == "wait":
            sample_row["native"][sample] = row
            sample_row["native_row"] = row
        else:
            target = action["target"]
            step = int(action["intervention_step"])
            sample_row["branches"][target][step] = row
    return by_sample


def utility(row: dict, tau_s: float, w_struct: float, w_cost: float, cost_ref: float) -> float | None:
    if not row.get("has_scores"):
        return None
    binding = float(row["binding"])
    structure = float(row["structure"])
    cost = float(row.get("compute") or 0.0)
    penalty = w_struct * max(0.0, tau_s - structure) ** 2
    return binding - penalty - w_cost * (cost / max(cost_ref, 1e-9))


def run_analysis(matrix: dict, tau_s: float, w_struct: float, w_cost: float) -> dict:
    result: dict = {"samples": {}, "per_target": {}, "cohort": defaultdict(list)}
    for sample, srow in matrix.items():
        native = srow.get("native_row")
        native_ok = bool(native and native.get("has_scores"))
        cost_ref = 1.0
        for target in TARGETS:
            branch = srow["branches"][target]
            points = []
            for step in STEPS:
                row = branch.get(step)
                if row is None or not row.get("has_scores"):
                    continue
                u = utility(row, tau_s, w_struct, w_cost, cost_ref)
                points.append(
                    {
                        "step": step,
                        "binding": float(row["binding"]),
                        "structure": float(row["structure"]),
                        "leakage": float(row.get("leakage") or 0.0),
                        "artifact": float(row.get("artifact") or 0.0),
                        "compute": float(row.get("compute") or 0.0),
                        "utility": u,
                        "eligible": u is not None,
                    }
                )
            native_point = None
            if native_ok:
                native_point = {
                    "step": -1,  # marker: no intervention
                    "binding": float(native["binding"]),
                    "structure": float(native["structure"]),
                    "leakage": float(native.get("leakage") or 0.0),
                    "artifact": float(native.get("artifact") or 0.0),
                    "compute": 0.0,
                    "utility": utility(native, tau_s, w_struct, w_cost, cost_ref),
                    "eligible": True,
                }
            eligible = [p for p in points if p["eligible"]]
            if native_point and native_point["eligible"]:
                eligible.append(native_point)
            best = max(eligible, key=lambda p: p["utility"]) if eligible else None
            result["samples"].setdefault(sample, {})[target] = {
                "points": points,
                "native": native_point,
                "t_star": best["step"] if best else None,
                "u_star": best["utility"] if best else None,
                "binding_star": best["binding"] if best else None,
                "structure_star": best["structure"] if best else None,
            }
            transition = srow.get("native_row", {}).get("transition") or "unknown"
            result["cohort"][transition].append(result["samples"][sample][target])
    result["per_target"] = {t: [v[t] for v in result["samples"].values()] for t in TARGETS}
    return result


def verdict_stats(analysis: dict) -> dict:
    """H2 verdict: spread of t* and the best-fixed vs per-sample-best gap."""
    stats: dict = {}
    for target, entries in analysis["per_target"].items():
        t_stars = [e["t_star"] for e in entries if e["t_star"] is not None]
        if not t_stars:
            stats[target] = {"n": 0}
            continue
        t_arr = np.asarray(t_stars, dtype=float)
        counts = {int(t): int((t_arr == t).sum()) for t in np.unique(t_arr)}
        # best fixed timing = mode over samples (max utility on average)
        fixed_utility = {
            t: float(np.nanmean([e["u_star"] for e in entries if e["t_star"] == t]))
            for t in np.unique(t_arr)
        }
        best_fixed_t = max(fixed_utility, key=fixed_utility.get)
        per_sample_best = float(np.nanmean([e["u_star"] for e in entries if e["t_star"] is not None]))
        fixed_achieved = float(
            np.nanmean(
                [
                    next((p["utility"] for p in e["points"] if p["step"] == int(best_fixed_t)), float("nan"))
                    for e in entries
                    if e["t_star"] is not None
                ]
            )
        )
        step0 = float(
            np.nanmean(
                [
                    next((p["utility"] for p in e["points"] if p["step"] == 0), float("nan"))
                    for e in entries
                    if e["t_star"] is not None
                ]
            )
        )
        stats[target] = {
            "n": len(t_stars),
            "t_star_hist": counts,
            "t_star_std": float(np.std(t_arr)),
            "t_star_range": [int(t_arr.min()), int(t_arr.max())],
            "distinct_t_star_bins": len(counts),
            "best_fixed_t": int(best_fixed_t),
            "per_sample_best_utility": per_sample_best,
            "best_fixed_utility": fixed_achieved,
            "step0_utility": step0,
            "adaptive_gain_over_fixed": per_sample_best - fixed_achieved,
            "fixed_gain_over_step0": fixed_achieved - step0,
        }
    return stats


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------

def plot_per_sample(matrix: dict, analysis: dict, out: Path) -> None:
    samples = sorted(analysis["samples"])
    n = len(samples)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
    style = {"A": "#d62728", "B": "#1f77b4", "AB": "#2ca02c"}
    for idx, sample in enumerate(samples):
        ax = axes[idx // cols][idx % cols]
        for target, entry in analysis["samples"][sample].items():
            color = style[target]
            ts = [p["step"] for p in entry["points"]]
            bs = [p["binding"] for p in entry["points"]]
            ss = [p["structure"] for p in entry["points"]]
            us = [p["utility"] if p["eligible"] else None for p in entry["points"]]
            ax.plot(ts, bs, color=color, marker="o", ms=3, lw=1.2, label=f"{target} bind")
            ax.plot(ts, ss, color=color, marker="s", ms=3, lw=0.9, ls="--", alpha=0.6, label=f"{target} struct")
            ax.plot(ts, us, color=color, marker="^", ms=3, lw=0.9, ls=":", alpha=0.9, label=f"{target} util")
            native = entry["native"]
            if native:
                ax.axhline(native["binding"], color=color, lw=0.8, alpha=0.35, ls="-.")
        ax.set_title(sample, fontsize=8)
        ax.set_xlabel("intervention step (0..24, -1=native)", fontsize=6)
        ax.set_ylabel("score", fontsize=6)
        ax.tick_params(labelsize=6)
        if idx == 0:
            ax.legend(fontsize=5, loc="upper right", ncol=3)
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle("H2: per-sample timing-quality curves (t=-1 is no intervention)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "h2_per_sample_curves.png", dpi=150)
    plt.close(fig)


def plot_histogram(analysis: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(4.2 * len(TARGETS), 3.2), squeeze=False)
    bins = np.arange(-2, 27, 2)
    for ax, target in zip(axes[0], TARGETS):
        t_stars = [e["t_star"] for e in analysis["per_target"][target] if e["t_star"] is not None]
        ax.hist(t_stars, bins=bins, color="#4c72b0", edgecolor="white")
        ax.set_title(f"target {target}: argmax timing (n={len(t_stars)})", fontsize=9)
        ax.set_xlabel("best intervention step (-1 = no intervention)", fontsize=7)
        ax.set_ylabel("samples", fontsize=7)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "h2_t_star_histogram.png", dpi=150)
    plt.close(fig)


def plot_aggregate(analysis: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(4.2 * len(TARGETS), 3.2), squeeze=False)
    for ax, target in zip(axes[0], TARGETS):
        entries = analysis["per_target"][target]
        mean_b, ci_b, mean_s, ci_s, mean_u, ci_u = [], [], [], [], [], []
        for step in STEPS:
            bs = [p["binding"] for e in entries for p in e["points"] if p["step"] == step]
            ss = [p["structure"] for e in entries for p in e["points"] if p["step"] == step]
            us = [p["utility"] for e in entries for p in e["points"] if p["step"] == step and p["eligible"]]
            if not bs:
                continue
            mean_b.append(np.mean(bs)); ci_b.append(1.96 * np.std(bs) / np.sqrt(len(bs)))
            mean_s.append(np.mean(ss)); ci_s.append(1.96 * np.std(ss) / np.sqrt(len(ss)))
            if us:
                mean_u.append(np.mean(us)); ci_u.append(1.96 * np.std(us) / np.sqrt(len(us)))
            else:
                mean_u.append(np.nan); ci_u.append(np.nan)
        xs = STEPS[: len(mean_b)]
        ax.errorbar(xs, mean_b, yerr=ci_b, marker="o", ms=4, capsize=2, label="binding")
        ax.errorbar(xs, mean_s, yerr=ci_s, marker="s", ms=4, capsize=2, ls="--", label="structure")
        ax.errorbar(xs, mean_u, yerr=ci_u, marker="^", ms=4, capsize=2, ls=":", label="utility")
        ax.set_title(f"target {target}: aggregate mean ± 95%CI", fontsize=9)
        ax.set_xlabel("intervention step", fontsize=7)
        ax.set_ylabel("score", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "h2_aggregate_curves.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def readiness_report(tasks: list[dict], sidecars: dict, overlay: dict) -> None:
    n_total = len(tasks)
    n_side = sum(1 for t in tasks if t["replay_id"] in sidecars)
    n_scored = sum(
        1 for t in tasks if has_scores({**sidecars.get(t["replay_id"], {}), **overlay.get(t["replay_id"], {})})
    )
    print("=" * 72)
    print("DATA READINESS")
    print("=" * 72)
    print(f"tasks in manifest        : {n_total}")
    print(f"with sidecar records     : {n_side}")
    print(f"with outcome scores      : {n_scored}")
    if n_scored == 0:
        print("BLOCKER: no C1/C2 outcome scores yet.")
        print("  -> run collect_counterfactuals.py on the remote GPU (SD3.5 weights),")
        print("     then fill binding_score / structure_score (see score contract in the")
        print("     collector sidecars) or provide an overlay JSONL via --scores.")
    print("-" * 72)
    header = ("sample", "transition", "native_correct", "lsda_step0_correct", "n_branches")
    rows = []
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        by_sample[task["sample_id"]].append(task)
    for sample in sorted(by_sample):
        group = by_sample[sample]
        meta = group[0].get("metadata", {})
        rows.append(
            (
                sample,
                group[0].get("source_transition"),
                meta.get("native_correct"),
                meta.get("lsda_step0_correct"),
                len(group),
            )
        )
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    for row in rows:
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, widths)))
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sidecars-root", type=Path, default=None)
    parser.add_argument("--scores", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("h2_analysis"))
    parser.add_argument("--structure-gate", type=float, default=0.5)
    parser.add_argument("--w-struct", type=float, default=1.0)
    parser.add_argument("--w-cost", type=float, default=0.1)
    args = parser.parse_args()

    payload = load_manifest(args.manifest)
    tasks: list[dict] = payload["tasks"]
    sidecars = load_sidecar_roots(args.sidecars_root) if args.sidecars_root else {}
    overlay = load_score_overlay(args.scores)

    readiness_report(tasks, sidecars, overlay)

    matrix = build_matrix(tasks, sidecars, overlay)
    scored_samples = sum(1 for s in matrix.values() if any(r.get("has_scores") for tgt in s["branches"].values() for r in tgt.values()))
    if scored_samples == 0:
        print("No outcome scores available -> H2 analysis cannot run yet. Aborting (exit 0).")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    analysis = run_analysis(matrix, args.structure_gate, args.w_struct, args.w_cost)
    stats = verdict_stats(analysis)

    print("\n" + "=" * 72)
    print("H2 VERDICT STATS (utility = binding - struct penalty - cost)")
    print("=" * 72)
    for target, s in stats.items():
        if s.get("n", 0) == 0:
            print(f"[{target}] no scored samples")
            continue
        print(f"[{target}] n={s['n']}  t* hist={s['t_star_hist']}  std={s['t_star_std']:.2f}  "
              f"distinct={s['distinct_t_star_bins']}")
        print(f"          per-sample-best util={s['per_sample_best_utility']:.3f} | "
              f"best-fixed(t={s['best_fixed_t']})={s['best_fixed_utility']:.3f} | step0={s['step0_utility']:.3f}")
        print(f"          adaptive gain over fixed={s['adaptive_gain_over_fixed']:+.3f} | "
              f"fixed gain over step0={s['fixed_gain_over_step0']:+.3f}")
    print("=" * 72)
    print("H2 INTERPRETATION")
    print("  * std / distinct bins of t* close to 0  -> optimal timing is (nearly) fixed;")
    print("    a fixed scheduler suffices and the agent narrative weakens.")
    print("  * adaptive_gain_over_fixed > 0         -> per-sample timing adds value.")
    print("  * fixed_gain_over_step0 > 0            -> timing matters even at fixed policy.")

    plot_per_sample(matrix, analysis, args.out_dir)
    plot_histogram(analysis, args.out_dir)
    plot_aggregate(analysis, args.out_dir)

    summary = {"args": {k: str(v) for k, v in vars(args).items()}, "verdict_stats": stats}
    for target in TARGETS:
        summary.setdefault("per_sample", {})
    for sample, by_target in analysis["samples"].items():
        summary["per_sample"][sample] = {
            t: {"t_star": v["t_star"], "u_star": v["u_star"]} for t, v in by_target.items()
        }
    (args.out_dir / "h2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nplots + summary written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
