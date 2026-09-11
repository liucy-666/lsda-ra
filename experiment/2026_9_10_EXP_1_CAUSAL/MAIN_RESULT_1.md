# Main Result 1（正式版）

> 对应图：Fig 1 `fig_probe_direction_aware.png`（主证据）、Fig 2 `fig_interaction_2x2.png`（终局验证）
> 数据：`deepdive/probe_v2/`（5 对 × 33 seeds @1024）、`e2_analysis/`（pair 1 × 9 seeds @512）

---

## 结论（一句话）

> **过去的工作把文化组合生图中的属性漂移归因于注意力的路由（Attention Weight $W$）；我们给出一个反直觉的结论：在文化组合生图场景下，注意力写回的“内容”（Value $V$）对错误的贡献远高于路由。**

**注意用词**：$W$ 与 $V$ 同属注意力机制。我们没有推翻“注意力是问题所在”，而是**在注意力内部把病灶从路由（$W$）重新定位到内容（$V$）**。

---

## 1. 前人归因：问题在“路由”

一系列工作认为，多实体属性漂移源于 **Attention Weight（$W=\mathrm{softmax}(QK^\top/\sqrt{d})$）**——即模型“看错了对象、看错了地方”，因此用注意力 mask、重加权、binding loss 来修**路由**。

## 2. 我们的反直觉发现：问题主要在“内容”

在同一次前向内，我们分别用干净 donor 分支的**路由**与**内容**替换目标区域，度量各自“朝正确表示补上的差距比例”：

$$\text{proj}_W=\frac{\langle O_{\text{wfix}}-O_{\text{mixed}},\Delta\rangle}{\lVert\Delta\rVert^2},\qquad
\text{proj}_V=\frac{\langle O_{\text{vfix}}-O_{\text{mixed}},\Delta\rangle}{\lVert\Delta\rVert^2},\qquad
\Delta=O_{\text{donor}}-O_{\text{mixed}}.$$

**结果（5 对文化实体，N=33 seeds，1024×1024）**：

| 指标 | 值 |
|---|---|
| `proj_V`（内容） | **0.82 ± 0.76** |
| `proj_W`（路由） | **0.23 ± 0.85** |
| 比值 | **≈ 3.6×** |

- 换**内容**平均补上 **82%** 的差距；换**路由**仅 **23%**。
- **5 对中 4 对一致**（pair_013 为异常样本）。
- 逐 seed 散点中，绝大多数点位于对角线**上方**（内容更有效）。

**Fig 1**：`figures/fig_probe_direction_aware.png`

## 3. 终局图验证：单独修任一因素均不足

在终局图像上做 2×2 因子（pair 1, N=9, 单点 L36/t26）：

| 臂 | 恢复量 | 翻转 | sign p |
|---|---|---|---|
| W-fix（只修路由） | +0.083 | 5/9 | 0.50（n.s.） |
| V-fix（只修内容） | +0.089 | 6/9 | 0.25（n.s.） |
| **Both-fix** | **+0.172** | **8/9** | **0.02** |

→ 单独修任一项均不显著；**二者需同时修正才稳定翻转**。

**Fig 2**：`figures/fig_interaction_2x2.png`

## 4. 必须写明的边界

1. **“正确方向”是相对定义**：以 donor（同前向 B-only 分支）为参照，非独立真值；因此只采信满足纳入门槛（A✓ B✓ SS✗）的样本。
2. **表示层 vs 终局**：`proj_V=0.82` 衡量**单个 (层,步)、右区表示**的差距闭合；终局图像要求全轨迹正确，故“内容主导表示差异”与“单点修内容不足以修终局”并不矛盾。
3. **可加性是均值巧合**：0.083+0.089=0.172 仅在均值成立，逐 seed 交互项正负抵消（0.00 ± 0.05），**不能据此推断 $W$ 与 $V$ 正交**（二者结构上同源）。
4. **样本与泛化**：5 对文化、33 seeds；pair_013 异常；需扩样本与跨文化对验证。

---

## 5. 与前人工作的关系（递进而非否定）

| | 前人 | 本文 |
|---|---|---|
| 归因 | 注意力 **路由** $W$ | 注意力 **内容** $V$ |
| 手段 | mask / 重加权 / binding loss | 内容级干预（如区域内容替换） |
| 关系 | —— | **在注意力内部重新定位病灶**，解释为何路由级方法有上限 |

---

## 6. 可直接放入论文的英文版

> **Main Result 1.** Prior work attributes cultural attribute drift in compositional generation to attention *routing* (the softmax weights $W$). We report a counter-intuitive finding: in cultural compositional generation, the *content* written back by attention (the value $V$) contributes far more to the error than the routing. Using a direction-aware interchange probe over five cultural entity pairs (N=33 seeds), fixing the content closes **82%** of the gap to the correct representation versus only **23%** for fixing the routing (**≈3.6×**). On the final image, fixing either factor alone is not significant; only fixing both reliably flips the binding (8/9, p=0.02). Since both $W$ and $V$ are components of attention, our result does not reject “attention” as the locus — it relocates the failure *within* attention from routing to content, and explains the structural ceiling of routing-level repairs.
