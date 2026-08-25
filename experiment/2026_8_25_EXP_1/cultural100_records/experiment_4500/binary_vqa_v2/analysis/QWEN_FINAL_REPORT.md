# Qwen 1800/1800 终版数据报告

## 数据范围

- `[data-supported]` Qwen 有效盲评：**1800/1800**。
- 独立图像：原生 SS **900** 张，LSDA **900** 张；二者按 `sample_id` 一一配对。
- 设计层级：100 个文化 Pair × 3 个 seed group × 3 个 replicate；每张组合图独立计分。
- 判定：左侧更像 standalone A 且右侧更像 standalone B 才算成功；任一侧错配即记属性漂移。
- 本报告只使用 Qwen；它不是 Qwen+Gemini 双盲共识终版。

## 核心计数

| 条件 | 成功图片 | 失败/漂移图片 | 总数 | 漂移率 | 95% Wilson CI |
|---|---:|---:|---:|---:|---:|
| 原生 SS | 633 | 267 | 900 | 29.67% | 26.77%–32.73% |
| LSDA | 853 | 47 | 900 | 5.22% | 3.95%–6.88% |

## 同图配对转移

| 原生 SS → LSDA | 图片数 |
|---|---:|
| 失败 → 成功（恢复） | 230 |
| 成功 → 失败（变坏） | 10 |
| 持续失败 | 37 |
| 持续成功 | 623 |

- `[data-supported]` 原生 SS 的 267 张失败图中，LSDA 恢复 **230 张（86.14%）**，仍失败 **37 张（13.86%）**。
- `[data-supported]` 原生 SS 的 633 张成功图中，LSDA 使 **10 张（1.58%）**转为失败。
- `[data-supported]` LSDA 相对原生 SS 的漂移率降低 **82.40%**；绝对差为 **-24.44 个百分点**，按文化 Pair 聚类 bootstrap 95% CI 为 **-28.89 至 -20.22 个百分点**。
- `[data-supported]` 配对方向检验：恢复 230 对、变坏 10 对，exact McNemar/binomial **p=1.71e-55**；恢复/变坏配对优势比 **23.0**（近似 95% CI 12.2–43.3）。

## 泄漏方向

| 条件 | A→B | B→A | 双向同时发生 |
|---|---:|---:|---:|
| 原生 SS | 200 | 204 | 137 |
| LSDA | 32 | 41 | 26 |

## Pair 层级稳定性

- `[data-supported]` 100 个 Pair 中：LSDA 漂移数减少 **78** 个、相同 **19** 个、增加 **3** 个。
- `[data-supported]` LSDA 在 **65** 个 Pair 的 9 张图中实现零漂移；原生 SS 对应为 **16** 个 Pair。

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
