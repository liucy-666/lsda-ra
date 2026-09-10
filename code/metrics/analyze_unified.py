"""Aggregate automated binding scores into the EXP_1-style statistics.

Computes (per metric and per majority-vote consensus):
  - drift rate per condition (native_SS / lsda_clean) with Wilson 95% CI
  - SS -> LSDA paired transitions (restored / worsened / persistent_failure / both_correct)
  - cluster bootstrap (resample cultural pairs) 95% CI of drift-rate difference
  - exact McNemar (restored vs worsened)
  - Cohen's kappa vs existing Qwen / Gemini blind ratings

Outputs:
  - experiment/2026_8_31_EXP_2/analysis/unified_summary.json
  - experiment/2026_8_31_EXP_2/report/UNIFIED_REPORT.md
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(r"D:\Python\MMDIT")
EXP = ROOT / "experiment" / "2026_8_31_EXP_2"
SCORES = EXP / "ratings" / "automated" / "binding_scores.jsonl"
VQA_ROOT = (
    ROOT
    / "experiment"
    / "2026_8_25_EXP_1"
    / "cultural100_records"
    / "experiment_4500"
    / "binary_vqa_v2"
)
BOOT_SEED = 20260831
BOOT_N = 20_000

METHODS = ["siglip", "clip_i", "clip_t", "dino"]


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def percentile(xs, p):
    xs = sorted(xs)
    pos = (len(xs) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def load_scores() -> list[dict]:
    if not SCORES.exists():
        return []
    rows = []
    # merge optional CSD scores by eval_id
    csd_map: dict[str, dict] = {}
    csd_path = SCORES.parent / "csd_scores.jsonl"
    if csd_path.exists():
        for line in csd_path.read_text(encoding="utf-8").splitlines():
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            if c.get("eval_id"):
                csd_map[c["eval_id"]] = c
    for line in SCORES.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        c = csd_map.get(rec.get("eval_id"))
        if c:
            for k, v in c.items():
                if k != "eval_id":
                    rec[k] = v
        rows.append(rec)
    return rows


def side_attr(rec: dict, method: str) -> tuple[str, str]:
    """Return (left_choice, right_choice) in {A, B} for a method."""
    pre = {
        "siglip": "siglip_cp",
        "clip_i": "clip_i",
        "clip_t": "clip_t",
        "dino": "dino",
        "csd": "csd",
    }[method]
    L_A, L_B = rec[f"{pre}_L_A"], rec[f"{pre}_L_B"]
    R_A, R_B = rec[f"{pre}_R_A"], rec[f"{pre}_R_B"]
    left = "A" if L_A >= L_B else "B"
    right = "A" if R_A >= R_B else "B"
    return left, right


def blip_side_attr(rec: dict) -> tuple[str | None, str | None]:
    bl = rec.get("blip", {})
    lA, lB = bl.get("L_A", ""), bl.get("L_B", "")
    rA, rB = bl.get("R_A", ""), bl.get("R_B", "")
    l = "A" if lA.startswith("yes") and not lB.startswith("yes") else ("B" if lB.startswith("yes") and not lA.startswith("yes") else None)
    r = "A" if rA.startswith("yes") and not rB.startswith("yes") else ("B" if rB.startswith("yes") and not rA.startswith("yes") else None)
    return l, r


def consensus_attr(rec: dict, methods: list[str]) -> tuple[str, str]:
    votes = defaultdict(int)
    for m in methods:
        if m == "blip":
            l, r = blip_side_attr(rec)
        else:
            l, r = side_attr(rec, m)
        if l:
            votes[f"L_{l}"] += 1
        if r:
            votes[f"R_{r}"] += 1
    left = "A" if votes["L_A"] > votes["L_B"] else "B"
    right = "A" if votes["R_A"] > votes["R_B"] else "B"
    return left, right


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    both = sum(1 for x, y in zip(a, b) if x == y)
    p_o = both / n
    p_a = sum(a) / n
    p_b = sum(b) / n
    p_e = p_a * p_b + (1 - p_a) * (1 - p_b)
    if p_e == 1:
        return float("nan")
    return (p_o - p_e) / (1 - p_e)


def main() -> None:
    scores = load_scores()
    print(f"scores loaded: {len(scores)}")
    has_scores = bool(scores)

    # existing VLM ratings keyed by eval_id
    def load_ratings(rater: str) -> dict[str, dict]:
        out = {}
        for p in sorted((VQA_ROOT / "ratings" / rater).glob("ratings_*.jsonl")):
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("eval_id") and r["eval_id"] not in out:
                    out[r["eval_id"]] = r
        return out

    qwen = load_ratings("QWEN")
    gemini = load_ratings("GEMINI")

    blind = json.loads((VQA_ROOT / "key" / "blind_map.json").read_text(encoding="utf-8"))
    bm = {b["eval_id"]: b for b in blind}

    summary: dict = {"metrics": {}, "consensus": {}}
    metric_keys = []

    if has_scores:
        metric_keys = METHODS
        if any("csd_L_A" in s for s in scores):
            metric_keys = metric_keys + ["csd"]
        if any("blip" in s for s in scores):
            metric_keys = metric_keys + ["blip"]

        for method in metric_keys:
            # per-image correctness
            correct = {}
            for rec in scores:
                if method == "blip":
                    bl = rec.get("blip", {})
                    ok = (
                        bl.get("L_A", "").startswith("yes") and not bl.get("L_B", "").startswith("yes")
                        and bl.get("R_B", "").startswith("yes") and not bl.get("R_A", "").startswith("yes")
                    )
                    correct[rec["eval_id"]] = bool(ok)
                else:
                    l, r = side_attr(rec, method)
                    correct[rec["eval_id"]] = (l == "A" and r == "B")
            stats = compute_stats(scores, correct)
            summary["metrics"][method] = stats

        # consensus (majority vote over the 4 core embedding methods; blip adds a 5th if present)
        # consensus = majority vote over core embedding metrics + CSD (blip excluded: broken)
        cons_methods = METHODS
        if "csd" in metric_keys:
            cons_methods = cons_methods + ["csd"]
        cons_correct = {}
        for rec in scores:
            l, r = consensus_attr(rec, cons_methods)
            cons_correct[rec["eval_id"]] = (l == "A" and r == "B")
        summary["consensus"] = compute_stats(scores, cons_correct)
        summary["consensus_methods"] = cons_methods
    else:
        cons_correct = {}
        cons_methods = []

    # kappa vs VLM ratings
    q_correct = {eid: bool(r.get("correct_binding")) for eid, r in qwen.items()}
    g_correct = {eid: bool(r.get("correct_binding")) for eid, r in gemini.items()}
    kvs = [(cons_correct[eid], q_correct[eid]) for eid in cons_correct if eid in q_correct]
    summary["kappa_vs_qwen"] = {
        "n": len(kvs),
        "cohen_kappa": kappa([x for x, _ in kvs], [y for _, y in kvs]),
    }
    kvs_g = [(cons_correct[eid], g_correct[eid]) for eid in cons_correct if eid in g_correct]
    summary["kappa_vs_gemini"] = {
        "n": len(kvs_g),
        "cohen_kappa": kappa([x for x, _ in kvs_g], [y for _, y in kvs_g]),
    }
    # Qwen vs Gemini reliability (uses existing ratings only)
    qg = [(q_correct[eid], g_correct[eid]) for eid in q_correct if eid in g_correct]
    summary["kappa_qwen_vs_gemini"] = {
        "n": len(qg),
        "cohen_kappa": kappa([x for x, _ in qg], [y for _, y in qg]),
    }
    # EXP_1 unfinished: dual-VLM consensus (both fail = failure) using existing ratings
    dual_correct = {}
    dual_rows = []
    for eid in q_correct:
        if eid in g_correct:
            b = bm.get(eid)
            if b is None:
                continue
            ok = q_correct[eid] and g_correct[eid]
            dual_correct[eid] = ok
            dual_rows.append(
                {
                    "eval_id": eid,
                    "sample_id": b["sample_id"],
                    "pair_id": b["pair_id"],
                    "condition": b["condition"],
                }
            )
    if dual_rows:
        summary["dual_vlm_consensus"] = compute_stats(dual_rows, dual_correct)
        # kappa between automated consensus and dual-VLM failure
        kvs_b = [(cons_correct[eid], not dual_correct[eid]) for eid in cons_correct if eid in dual_correct]
        summary["kappa_vs_dual_vlm_fail"] = {
            "n": len(kvs_b),
            "cohen_kappa": kappa([x for x, _ in kvs_b], [y for _, y in kvs_b]),
        }

    (EXP / "analysis").mkdir(parents=True, exist_ok=True)
    (EXP / "analysis" / "unified_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # per-image source CSV for traceability (dual-VLM + single VLM + automated when present)
    import csv

    src = EXP / "source_data"
    src.mkdir(parents=True, exist_ok=True)
    with (src / "image_level.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            ["eval_id", "sample_id", "pair_id", "seed_group", "replicate", "latent_seed",
             "condition", "qwen_correct", "gemini_correct", "dual_vlm_correct",
             "automated_consensus_correct"]
        )
        for b in blind:
            eid = b["eval_id"]
            q = q_correct.get(eid)
            g = g_correct.get(eid)
            dual = dual_correct.get(eid)
            auto = cons_correct.get(eid)
            w.writerow(
                [eid, b["sample_id"], b["pair_id"], b["seed_group"], b["replicate"],
                 b["latent_seed"], b["condition"],
                 "" if q is None else int(q), "" if g is None else int(g),
                 "" if dual is None else int(dual), "" if auto is None else int(auto)]
            )

    # report
    (EXP / "report").mkdir(parents=True, exist_ok=True)
    lines = ["# 统一自动化评估报告（LSDA clean v1 vs native SS）", ""]
    lines.append(f"- 数据：EXP_1 的 900+900 同种子图像（100 pair × 3 seed × 3 replicate）")
    lines.append(f"- 自动化指标：{'、'.join(cons_methods) if cons_methods else '(待打分)'}（MaSC masked-maxcos / CLIP / DINO / CSD / BLIP-VQA）")
    lines.append(f"- 与 EXP_1 已有 VLM 盲评的一致性：见文末 κ 表")
    lines.append("")
    blocks = [("consensus(automated)", summary["consensus"])]
    blocks += [(f"metric:{m}", summary["metrics"][m]) for m in metric_keys]
    if "dual_vlm_consensus" in summary:
        blocks.append(("dual_vlm_consensus(both-fail=fail)", summary["dual_vlm_consensus"]))
    for label, block in blocks:
        if not block:
            continue
        lines.append(f"## {label}")
        for cond in ("native_SS", "lsda_clean"):
            st = block[cond]
            lines.append(
                f"- {cond}: 漂移率 {st['drift_rate']:.2%} ({st['drift']}/{st['n']}) "
                f"95%CI [{st['wilson95'][0]:.2%}, {st['wilson95'][1]:.2%}]"
            )
        tr = block["transitions"]
        lines.append(
            f"- 转移: 恢复 {tr['restored']} / 变坏 {tr['worsened']} / 持续失败 {tr['persistent_failure']} / 持续成功 {tr['both_correct']}"
        )
        lines.append(f"- McNemar p={block['mcnemar_p']:.3g}; 相对漂移降低 {block['relative_drift_reduction']:.1%}")
        lines.append("")
    lines.append("## 与已有 VLM 盲评的一致性 (Cohen's kappa)")
    lines.append(
        f"- vs Qwen: κ={summary['kappa_vs_qwen']['cohen_kappa']:.3f} (n={summary['kappa_vs_qwen']['n']})"
    )
    lines.append(
        f"- vs Gemini: κ={summary['kappa_vs_gemini']['cohen_kappa']:.3f} (n={summary['kappa_vs_gemini']['n']})"
    )
    lines.append(
        f"- Qwen vs Gemini 互评: κ={summary['kappa_qwen_vs_gemini']['cohen_kappa']:.3f} (n={summary['kappa_qwen_vs_gemini']['n']})"
    )
    if "kappa_vs_dual_vlm_fail" in summary:
        lines.append(
            f"- vs 双VLM均判失败: κ={summary['kappa_vs_dual_vlm_fail']['cohen_kappa']:.3f} (n={summary['kappa_vs_dual_vlm_fail']['n']})"
        )
    (EXP / "report" / "UNIFIED_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=1)[:3000])


def compute_stats(scores: list[dict], correct: dict[str, bool]):
    by_cond: dict[str, list[dict]] = defaultdict(list)
    by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for rec in scores:
        eid = rec["eval_id"]
        ok = correct.get(eid, False)
        rec = {**rec, "_correct": ok}
        by_cond[rec["condition"]].append(rec)
        by_sample[rec["sample_id"]][rec["condition"]] = rec

    cond_stats = {}
    for cond in ("native_SS", "lsda_clean"):
        rows = by_cond.get(cond, [])
        n = len(rows)
        drift = sum(1 for r in rows if not r["_correct"])
        lo, hi = wilson(drift, n)
        cond_stats[cond] = {
            "n": n,
            "correct": n - drift,
            "drift": drift,
            "drift_rate": drift / n if n else None,
            "wilson95": [lo, hi],
        }

    transitions = Counter()
    for sid, d in by_sample.items():
        if "native_SS" not in d or "lsda_clean" not in d:
            continue
        ss, ls = d["native_SS"]["_correct"], d["lsda_clean"]["_correct"]
        transitions[
            "restored" if (not ss and ls)
            else "worsened" if (ss and not ls)
            else "persistent_failure" if (not ss and not ls)
            else "both_correct"
        ] += 1

    pair_groups: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for sid, d in by_sample.items():
        if "native_SS" in d and "lsda_clean" in d:
            pair_groups[d["native_SS"]["pair_id"]].append(
                (d["native_SS"]["_correct"], d["lsda_clean"]["_correct"])
            )

    rng = random.Random(BOOT_SEED)
    pids = sorted(pair_groups)
    boot_diff = []
    for _ in range(BOOT_N):
        sampled = [rng.choice(pids) for _ in pids]
        rows = [x for pid in sampled for x in pair_groups[pid]]
        d = sum(1 for ss, ls in rows if not ls) - sum(1 for ss, ls in rows if not ss)
        boot_diff.append(d / len(rows))
    diff = cond_stats["lsda_clean"]["drift_rate"] - cond_stats["native_SS"]["drift_rate"]
    b = transitions["restored"]
    c = transitions["worsened"]
    mcnemar = binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue if b + c else float("nan")

    return {
        "native_SS": cond_stats["native_SS"],
        "lsda_clean": cond_stats["lsda_clean"],
        "transitions": dict(transitions),
        "drift_rate_difference_lsda_minus_ss": diff,
        "cluster_bootstrap95": [percentile(boot_diff, 0.025), percentile(boot_diff, 0.975)],
        "relative_drift_reduction": (
            1 - cond_stats["lsda_clean"]["drift_rate"] / cond_stats["native_SS"]["drift_rate"]
            if cond_stats["native_SS"]["drift_rate"]
            else None
        ),
        "mcnemar_p": mcnemar,
        "restored": b,
        "worsened": c,
    }


if __name__ == "__main__":
    main()
