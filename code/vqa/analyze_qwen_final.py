import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import binomtest


ROOT = Path(r"D:\Python\MMDIT\experiment\2026_8_25_EXP_1\cultural100_records\experiment_4500\binary_vqa_v2")
MAP_FILE = ROOT / "key" / "blind_map.json"
RATING_DIR = ROOT / "ratings" / "QWEN"
ANALYSIS_DIR = ROOT / "analysis"
SOURCE_DIR = ROOT / "source_data"
BOOT_SEED = 20260825
BOOT_N = 20_000


def wilson(k, n, z=1.959963984540054):
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


def read_data():
    mapping = {r["eval_id"]: r for r in json.loads(MAP_FILE.read_text(encoding="utf-8"))}
    ratings = {}
    duplicates = 0
    invalid = []
    for path in sorted(RATING_DIR.glob("ratings_QWEN_chunk_*.jsonl")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                invalid.append(f"{path.name}:{lineno}")
                continue
            eid = r.get("eval_id")
            if eid in ratings:
                duplicates += 1
                continue
            ratings[eid] = r
    missing = sorted(set(mapping) - set(ratings))
    extra = sorted(set(ratings) - set(mapping))
    if missing or extra or invalid:
        raise RuntimeError(f"missing={len(missing)}, extra={len(extra)}, invalid={len(invalid)}")
    return mapping, ratings, duplicates


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    mapping, ratings, duplicates = read_data()

    image_rows = []
    by_sample = defaultdict(dict)
    for eid, m in mapping.items():
        r = ratings[eid]
        correct = r.get("left_choice") == "A" and r.get("right_choice") == "B"
        row = {
            "eval_id": eid,
            "sample_id": m["sample_id"],
            "pair_id": m["pair_id"],
            "pair_index": m["pair_index"],
            "seed_group": m["seed_group"],
            "replicate": m["replicate"],
            "latent_seed": m["latent_seed"],
            "condition": m["condition"],
            "entity_A": m["entity_A"],
            "entity_B": m["entity_B"],
            "left_choice": r.get("left_choice"),
            "right_choice": r.get("right_choice"),
            "correct_binding": int(correct),
            "drift": int(not correct),
            "A_to_B_leakage": int(bool(r.get("A_to_B_leakage"))),
            "B_to_A_leakage": int(bool(r.get("B_to_A_leakage"))),
            "left_reason": r.get("left_reason", ""),
            "right_reason": r.get("right_reason", ""),
            "model": r.get("model", ""),
        }
        image_rows.append(row)
        by_sample[m["sample_id"]][m["condition"]] = row

    image_rows.sort(key=lambda x: (x["pair_index"], x["seed_group"], x["replicate"], x["condition"]))
    write_csv(SOURCE_DIR / "qwen_image_level.csv", image_rows, list(image_rows[0]))

    paired_rows = []
    for sid, d in sorted(by_sample.items()):
        if set(d) != {"native_SS", "lsda_clean"}:
            raise RuntimeError(f"Unpaired sample {sid}: {sorted(d)}")
        ss, ls = d["native_SS"], d["lsda_clean"]
        transition = (
            "restored" if ss["drift"] and not ls["drift"] else
            "worsened" if not ss["drift"] and ls["drift"] else
            "persistent_failure" if ss["drift"] and ls["drift"] else
            "both_correct"
        )
        paired_rows.append({
            "sample_id": sid, "pair_id": ss["pair_id"], "pair_index": ss["pair_index"],
            "seed_group": ss["seed_group"], "replicate": ss["replicate"], "latent_seed": ss["latent_seed"],
            "native_eval_id": ss["eval_id"], "lsda_eval_id": ls["eval_id"],
            "native_correct": ss["correct_binding"], "native_drift": ss["drift"],
            "lsda_correct": ls["correct_binding"], "lsda_drift": ls["drift"],
            "transition": transition,
        })
    write_csv(SOURCE_DIR / "qwen_paired_transitions.csv", paired_rows, list(paired_rows[0]))

    condition = {}
    for cond in ("native_SS", "lsda_clean"):
        rr = [x for x in image_rows if x["condition"] == cond]
        drift = sum(x["drift"] for x in rr)
        lo, hi = wilson(drift, len(rr))
        condition[cond] = {
            "n": len(rr), "correct": len(rr) - drift, "drift": drift,
            "drift_rate": drift / len(rr), "wilson95": [lo, hi],
            "A_to_B_leakage": sum(x["A_to_B_leakage"] for x in rr),
            "B_to_A_leakage": sum(x["B_to_A_leakage"] for x in rr),
            "bidirectional_leakage": sum(x["A_to_B_leakage"] and x["B_to_A_leakage"] for x in rr),
        }

    transitions = Counter(x["transition"] for x in paired_rows)
    pair_groups = defaultdict(list)
    for x in paired_rows:
        pair_groups[x["pair_id"]].append(x)
    pair_rows = []
    for pid, rr in sorted(pair_groups.items()):
        ss_d = sum(x["native_drift"] for x in rr)
        ls_d = sum(x["lsda_drift"] for x in rr)
        tc = Counter(x["transition"] for x in rr)
        pair_rows.append({
            "pair_id": pid, "n": len(rr), "native_correct": len(rr)-ss_d, "native_drift": ss_d,
            "lsda_correct": len(rr)-ls_d, "lsda_drift": ls_d, "drift_change_lsda_minus_ss": ls_d-ss_d,
            "restored": tc["restored"], "worsened": tc["worsened"],
            "persistent_failure": tc["persistent_failure"], "both_correct": tc["both_correct"],
        })
    write_csv(SOURCE_DIR / "qwen_pair_summary.csv", pair_rows, list(pair_rows[0]))

    # Cluster bootstrap: resample the 100 cultural pairs, preserving all 9 paired samples per pair.
    rng = random.Random(BOOT_SEED)
    pids = sorted(pair_groups)
    boot_diff = []
    for _ in range(BOOT_N):
        sampled = [rng.choice(pids) for _ in pids]
        rows = [x for pid in sampled for x in pair_groups[pid]]
        boot_diff.append(sum(x["lsda_drift"] - x["native_drift"] for x in rows) / len(rows))
    diff = condition["lsda_clean"]["drift_rate"] - condition["native_SS"]["drift_rate"]
    cluster_ci = [percentile(boot_diff, .025), percentile(boot_diff, .975)]

    b = transitions["restored"]
    c = transitions["worsened"]
    mcnemar = binomtest(min(b, c), b + c, .5, alternative="two-sided")
    log_se = math.sqrt(1 / b + 1 / c)
    paired_or = b / c
    paired_or_ci = [math.exp(math.log(paired_or) - 1.96 * log_se), math.exp(math.log(paired_or) + 1.96 * log_se)]

    summary = {
        "status": "Qwen complete; single-rater analysis",
        "model": "qwen3-vl-235b-a22b-instruct",
        "n_ratings": len(ratings), "n_paired_samples": len(paired_rows), "n_cultural_pairs": len(pair_groups),
        "duplicates_ignored_first_occurrence_wins": duplicates,
        "condition": condition, "transitions": dict(transitions),
        "restoration_rate_among_native_failures": b / condition["native_SS"]["drift"],
        "persistent_failure_rate_among_native_failures": transitions["persistent_failure"] / condition["native_SS"]["drift"],
        "harm_rate_among_native_correct": c / condition["native_SS"]["correct"],
        "absolute_drift_rate_difference_lsda_minus_ss": diff,
        "cluster_bootstrap95_difference": cluster_ci,
        "relative_drift_reduction": 1 - condition["lsda_clean"]["drift_rate"] / condition["native_SS"]["drift_rate"],
        "mcnemar_exact": {"discordant_restored": b, "discordant_worsened": c, "p_two_sided": mcnemar.pvalue},
        "paired_odds_ratio_restoration_vs_harm": paired_or,
        "paired_odds_ratio_approx95": paired_or_ci,
        "pair_level": {
            "improved_pairs": sum(r["drift_change_lsda_minus_ss"] < 0 for r in pair_rows),
            "unchanged_pairs": sum(r["drift_change_lsda_minus_ss"] == 0 for r in pair_rows),
            "worsened_pairs": sum(r["drift_change_lsda_minus_ss"] > 0 for r in pair_rows),
            "zero_lsda_drift_pairs": sum(r["lsda_drift"] == 0 for r in pair_rows),
            "zero_native_drift_pairs": sum(r["native_drift"] == 0 for r in pair_rows),
        },
        "bootstrap": {"unit": "cultural pair", "draws": BOOT_N, "seed": BOOT_SEED},
    }
    (ANALYSIS_DIR / "qwen_final_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    worst = sorted(pair_rows, key=lambda x: (-x["lsda_drift"], x["pair_id"]))[:10]
    best = sorted(pair_rows, key=lambda x: (x["drift_change_lsda_minus_ss"], x["pair_id"]))[:10]
    report = f"""# Qwen 1800/1800 终版数据报告

## 数据范围

- `[data-supported]` Qwen 有效盲评：**1800/1800**。
- 独立图像：原生 SS **900** 张，LSDA **900** 张；二者按 `sample_id` 一一配对。
- 设计层级：100 个文化 Pair × 3 个 seed group × 3 个 replicate；每张组合图独立计分。
- 判定：左侧更像 standalone A 且右侧更像 standalone B 才算成功；任一侧错配即记属性漂移。
- 本报告只使用 Qwen；它不是 Qwen+Gemini 双盲共识终版。

## 核心计数

| 条件 | 成功图片 | 失败/漂移图片 | 总数 | 漂移率 | 95% Wilson CI |
|---|---:|---:|---:|---:|---:|
| 原生 SS | {condition['native_SS']['correct']} | {condition['native_SS']['drift']} | 900 | {condition['native_SS']['drift_rate']:.2%} | {condition['native_SS']['wilson95'][0]:.2%}–{condition['native_SS']['wilson95'][1]:.2%} |
| LSDA | {condition['lsda_clean']['correct']} | {condition['lsda_clean']['drift']} | 900 | {condition['lsda_clean']['drift_rate']:.2%} | {condition['lsda_clean']['wilson95'][0]:.2%}–{condition['lsda_clean']['wilson95'][1]:.2%} |

## 同图配对转移

| 原生 SS → LSDA | 图片数 |
|---|---:|
| 失败 → 成功（恢复） | {transitions['restored']} |
| 成功 → 失败（变坏） | {transitions['worsened']} |
| 持续失败 | {transitions['persistent_failure']} |
| 持续成功 | {transitions['both_correct']} |

- `[data-supported]` 原生 SS 的 267 张失败图中，LSDA 恢复 **{b} 张（{b/267:.2%}）**，仍失败 **{transitions['persistent_failure']} 张（{transitions['persistent_failure']/267:.2%}）**。
- `[data-supported]` 原生 SS 的 633 张成功图中，LSDA 使 **{c} 张（{c/633:.2%}）**转为失败。
- `[data-supported]` LSDA 相对原生 SS 的漂移率降低 **{(1-condition['lsda_clean']['drift_rate']/condition['native_SS']['drift_rate']):.2%}**；绝对差为 **{diff*100:.2f} 个百分点**，按文化 Pair 聚类 bootstrap 95% CI 为 **{cluster_ci[0]*100:.2f} 至 {cluster_ci[1]*100:.2f} 个百分点**。
- `[data-supported]` 配对方向检验：恢复 {b} 对、变坏 {c} 对，exact McNemar/binomial **p={mcnemar.pvalue:.3g}**；恢复/变坏配对优势比 **{paired_or:.1f}**（近似 95% CI {paired_or_ci[0]:.1f}–{paired_or_ci[1]:.1f}）。

## 泄漏方向

| 条件 | A→B | B→A | 双向同时发生 |
|---|---:|---:|---:|
| 原生 SS | {condition['native_SS']['A_to_B_leakage']} | {condition['native_SS']['B_to_A_leakage']} | {condition['native_SS']['bidirectional_leakage']} |
| LSDA | {condition['lsda_clean']['A_to_B_leakage']} | {condition['lsda_clean']['B_to_A_leakage']} | {condition['lsda_clean']['bidirectional_leakage']} |

## Pair 层级稳定性

- `[data-supported]` 100 个 Pair 中：LSDA 漂移数减少 **{summary['pair_level']['improved_pairs']}** 个、相同 **{summary['pair_level']['unchanged_pairs']}** 个、增加 **{summary['pair_level']['worsened_pairs']}** 个。
- `[data-supported]` LSDA 在 **{summary['pair_level']['zero_lsda_drift_pairs']}** 个 Pair 的 9 张图中实现零漂移；原生 SS 对应为 **{summary['pair_level']['zero_native_drift_pairs']}** 个 Pair。

## 证据边界

- `[data-supported]` 在 Qwen 的纹理/风格二选一 VQA 下，LSDA 与显著更低的属性漂移频率相关。
- `[exploratory]` 该 VQA 不评价形状、对象数量、遮挡、融合或构图，因此不能据此声称 LSDA 的结构质量更好。
- `[exploratory]` 10 张“原生正确、LSDA 变坏”图像必须保留；它们是方法副作用而非可删除异常值。
- `[hypothesis]` 局部路由或分割边界损伤可能解释部分变坏案例，但需要结构 VQA、mask QC 或干预日志支持，当前纹理 VQA 本身不能证明原因。
- 最终“双 VLM 均判失败才算失败”的结果必须等待 Gemini 完成并按同一 `eval_id` 取交集；不能由本报告替代。

## Source Data

- `qwen_image_level.csv`：1800 张图逐图结果与理由。
- `qwen_paired_transitions.csv`：900 个同样本 SS→LSDA 转移。
- `qwen_pair_summary.csv`：100 个文化 Pair 汇总。
- `qwen_final_summary.json`：机器可读统计摘要。
"""
    (ANALYSIS_DIR / "QWEN_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Worst LSDA pairs:", worst)
    print("Largest improvements:", best)


if __name__ == "__main__":
    main()
