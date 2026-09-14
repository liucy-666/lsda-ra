# Result 2 小规模验证结果 — KA/ME 分流与统一内容源修复

> 日期：2026-09-12
> 数据：100 cultural pairs × 3 seeds（1011/1012/1013）= 300 样本，1024×1024，28 步，CFG 4.5
> 裁判：GPT-5.4 + Gemini-3.5-flash，严格双裁（两侧都达标才算绑定正确）
> 产物：`figures/fig_ka_me_drift.png`、`analysis/metrics.json`、`analysis/classification.json`

## 1. 诊断分类（300 样本）

| 类别 | 数量 | 占比 |
|---|---:|---:|
| **KA 知识性缺失**（standalone A 或 B 失败）| 190 | 63.3% |
| **ME 机制性错误**（A✓ B✓ 但组合泄漏）| 60 | 20.0% |
| **BC 双正确** | 50 | 16.7% |

**要点：失败中 KA 占绝大多数（190/250=76%），ME 仅 60。** 这与"多数失败是知识缺失、而非组合串味"的假设一致。

## 2. 指标（M1 Drift / 错绑率，越低越好）

| 方法 | 成功 | Drift | 95% CI（pair-clustered bootstrap）|
|---|---:|---:|---|
| Native SS | 109/300 | **63.7%** | [56.0, 71.0] |
| LSDA uniform（不分流，全失败上 LSDA）| 156/300 | **48.0%** | [39.7, 56.3] |
| **Ours（routed：ME→LSDA，KA→LSDA+知识短语）** | 178/300 | **40.7%** | [33.3, 48.3] |

- Ours 相对 Native 降 **23.0 pts**；相对 uniform 再降 **7.3 pts**。
- uniform 的增益（−15.7 pts）几乎全部来自 ME 子集。
- 注：Wilson CI 分别为 Native [58.1,68.9]、uniform [42.4,53.6]、routed [35.3,46.3]；聚类后区间略宽，正文报聚类版。

## 3. 逐类分解

| 子集 | n | Native | Uniform | Routed |
|---|---:|---:|---:|---:|
| **ME** | 60 | 100.0% | 33.3% | 33.3% |
| **KA** | 190 | 68.9% | 65.3% | **53.7%** |

- **ME**：LSDA 修复 2/3（100%→33.3%）；知识注入对 ME 无额外作用（预期）。
- **KA**：不加知识几乎无效（68.9→65.3）；**加知识短语后降到 53.7%**（−15.2 pts），但仍高——一部分是"真知识缺失"，推理期不可修复。

## 4. 结构代价（M3，VLM structure 均值）

| 方法 | structure |
|---|---:|
| Native | 0.938 |
| Uniform | 0.722 |
| Routed | 0.746 |

任何修复都会降结构分（局部重写/矩形区域的代价）；Routed 略优于 Uniform。

## 5. 已完成的方法学配置

- **ME 修复**：LSDA v1.4 矩形区域（SAM 仅定位，`RECT_PAD=16`），专家读干净 short prompt。
- **KA 修复**：同一 LSDA，专家读 `short + 规则抽取的属性短语`（`manifests/kb_attrs.json`）。
- **诊断门**：standalone A/B + native SS，双裁 ≥0.5。
- 修复运行：438/440 成功（2 个 `Empty latent owner`，样本 `p073_s1012`）。

## 6. 必须写明的边界

1. **CI 为 pair-clustered bootstrap**（10k draws，按 pair 重采样）；Wilson 版见 §2 注。
2. **KA 占比高**可能受"standalone 判定用 short prompt + 双裁严格"影响，需与历史口径对照说明。
3. **结构分下降**是真实代价，需在正文报告，不能只报 drift。
4. 修复后的具体纹样与 standalone 仍不完全一致（LSDA 上限）。
5. 本实验未包含**路由级基线**（attention binding 需按 100 对泛化实现），列入下一步。

## 7. 结论一句话

> 在 100 对 × 3 seeds 上，**Native 错绑率 63.7%**；内容级 LSDA 降到 **48.0%**；先诊断再分流（ME 用 LSDA、KA 注入知识短语）进一步降到 **40.7%**。其中 ME 修复率 2/3，KA 仅靠知识短语可降 15.2 pts，剩余为训练无关不可修复的真知识缺失。

## 8. 客观指标（CLIP-I / CLIP-T / DINOv2）——补充分析

实现：`code/metrics/score_binding_census.py`（本地 CPU，单进程，逐行落盘）；图：`figures/fig_metrics_compare.png`；数据：`analysis/objective_metrics.json`。
做法：把组合图按中线切左右半区，与该对的 standalone A/B 参考图/短文本做余弦，按左右归属判正确。

| 方法 | VLM | CLIP-I | CLIP-T | DINOv2 |
|---|---:|---:|---:|---:|
| Native SS | 63.7% | 75.6% | 75.3% | 69.0% |
| LSDA uniform | 47.8% | 45.2% | 47.2% | 42.5% |
| Ours (routed) | **40.5%** | 49.2% | 49.5% | 45.8% |

**与 VLM 一致率**：CLIP-I ≈ 52–58%、CLIP-T ≈ 54–62%、DINOv2 ≈ 53–60%（接近随机）。

**判别间距**（mean |Δscore|，A vs B）：CLIP-I 0.049–0.053、CLIP-T 0.027–0.037、DINOv2 0.14–0.17，且存在左右偏差（L 正确率 77–82% vs R 58–68%）。

**结论（诚实）**：
1. 客观指标能大致反映"Native 高、修复后低"，但**系统性高估 Native**；
2. **无法体现"诊断分流"的增益**（Ours 在三个客观指标上反而略差于 uniform），与 VLM 结论相反；
3. 与 VLM 一致率仅略高于随机 → 细粒度文化绑定**超出通用图像编码器（CLIP/DINO）的判别能力**；这与项目历史结论（自动指标与 VLM κ≈−0.35）一致。
4. 因此：**VLM 双裁为正式主指标；客观指标仅作敏感性/对照**，并可作为"为什么需要细粒度文化评测"的论据。
