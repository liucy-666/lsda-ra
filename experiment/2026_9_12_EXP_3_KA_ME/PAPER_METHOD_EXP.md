# Method & Experiments（论文草稿）

> 实验：`experiment/2026_9_12_EXP_3_KA_ME/`；数据：100 pairs × 3 seeds = 300 samples。
> 指标主口径：GPT-5.4 + Gemini-3.5-flash 严格双裁；CI 为 pair-clustered bootstrap。
> 待补：路由级基线（`binding/` 正在生成），完成后替换 `TABLE 1` 中的 ATTENTION BINDING 行。

---
---

# 3. Method

## 3.1 Problem setup

We study **cultural attribute binding** in text-to-image generation: two cultural entities
A (left) and B (right) are requested in one scene, and the generated image may render one
entity with the other's cultural attributes (e.g., an Italian maiolica vase painted as
Chinese blue-and-white). We call this a *binding failure*. Given a scene prompt
`p_AB` and per-entity short prompts `p_A, p_B`, the goal is to produce a scene in which the
surface cultural attributes (material, palette, technique, motif) of each region match its
own entity.

## 3.2 Diagnosing knowledge absence vs. mechanism error

A binding failure has two mutually exclusive root causes, and they must not be conflated:

- **Mechanism error (ME).** The model can render A and B *alone*, but the composition
  contaminates one region with the other's content.
- **Knowledge absence (KA).** The model cannot render A and/or B even *alone*; the
  cultural concept is not accessible from its conditioning/weights.

We operationalize the diagnosis with a **standalone gate**. For each `(pair, seed)` we
generate standalone A, standalone B, and the native SS composition at resolution 1024.
Two independent VLMs (GPT-5.4 and Gemini-3.5-flash) score each image. With a strict
dual-rater rule (`matches >= 0.5`; binding correct iff `left→A` and `right→B` for **both**
raters):

```
KA : not A_ok or not B_ok
ME : A_ok and B_ok and SS leaks
BC : A_ok and B_ok and SS correct   (no intervention)
```

## 3.3 Mechanism-error repair: local re-diffusion (LSDA)

For ME samples we use **local re-diffusion**: each entity region is denoised by a
*region expert* conditioned only on that entity's short prompt, from step 0 to the end,
while the rest of the scene follows the frozen native SS trajectory.

**(a) Region definition.** A SAM [ref] pass on the native SS image localizes each entity;
we reduce each instance mask to its **padded bounding rectangle** (`pad = 16 px`). Using a
rectangle rather than a tight contour moves the expert's write boundary into the smooth
background, which removes the hard seam/white notch produced by a tight mask whose shape
disagrees with the expert's regenerated object.

**(b) Composition.** At every denoising step `t`,

```
z_{t+1} = Σ_i w_i ⊙ cand_i(z_t^{crop}, p_i)  +  w_bg ⊙ z_{t+1}^{native},
```

where `w_i` is the (anti-aliased, area-pooled) soft ownership of rectangle `i`,
`w_bg = 1 − Σ_i w_i`, `cand_i` is the region expert's scheduler update inside its crop,
and `z^{native}` is the frozen same-seed native SS trajectory. This keeps every pixel owned
(either by an entity expert or by the native background) and makes boundaries a soft blend.

## 3.4 Knowledge-absence repair: knowledge injection into the local expert

For KA samples the region expert is fed the same short prompt **plus rule-extracted
attribute phrases** derived from the entity's long description (material, palette,
technique, motif), e.g.:

```
a Chinese blue-and-white porcelain vase
  + "cobalt-blue underglaze painting; Chinese landscape; lotus; scrolling-cloud motifs"
```

The knowledge is thus injected at the *content source* of the same local expert — no new
module, no fine-tuning, and no reference-image adapter.

## 3.5 Routed pipeline

```
for each (pair, seed):
    generate standalone A, B, native SS
    diagnose (standalone gate)
    if ME:  local re-diffusion with short prompts
    if KA:  local re-diffusion with short prompts + attribute phrases
    if BC:  leave native SS unchanged
```

---
---

# 4. Experiments

## 4.1 Setup

- Model: SD3.5-large; 1024×1024; 28 steps; CFG 4.5; fp16.
- Data: 100 cultural entity pairs, 3 fixed seeds each (`1011,1012,1013`) → **300 samples**.
- Judges: GPT-5.4 + Gemini-3.5-flash, strict dual-rater; `reasoning_effort=none`.
- Projection CIs: pair-clustered bootstrap (10k draws), except where noted.

## 4.2 Metrics

| ID | Metric | Definition | Direction |
|---|---|---|---|
| **M1** | **Drift / mis-binding rate** | fraction of samples where `left→A ∧ right→B` fails for **both** judges | ↓ |
| M3 | Structure score | mean VLM structure score (1 = clean, two separate entities) | ↑ |
| — | Objective (secondary) | CLIP-I / CLIP-T / DINOv2 region attribution | ↓ |

> **M1 is the primary metric**; it is stricter than the single-judge / preference protocols
> common in related work. CLIP/DINO are reported only as a sensitivity analysis (Sec. 4.6).

## 4.3 Census: the failure population is mostly KA

| Category | n | share |
|---|---:|---:|
| Knowledge absence (KA) | 190 | 63.3% |
| Mechanism error (ME) | 60 | 20.0% |
| Both correct (BC) | 50 | 16.7% |

Among the 250 failures, **KA accounts for 76%**. Mitigation methods that do not separate
KA from ME therefore operate on a population that is dominated by a cause they do not model.

## 4.4 Main result

**TABLE 1. Same-dataset comparison (n = 300, 100 pairs × 3 seeds).**

| Method | Type | M1 Drift ↓ | 95% CI | M3 Structure ↑ |
|---|---|---:|---|---:|
| SD3.5-large (native SS) | — | 63.7% | [56.0, 71.0] | 0.938 |
| Attention binding (routing-level, DreamRenderer-style) | routing | *(pending)* | — | *(pending)* |
| LSDA (content-level, uniform) | content | 48.0% | [39.7, 56.3] | 0.722 |
| **Ours (routed: diagnose → LSDA / LSDA+knowledge)** | content + routing-by-diagnosis | **40.7%** | [33.3, 48.3] | 0.746 |

- Ours reduces drift by **23.0 points** vs. native SD3.5 and by **7.3 points** vs. the
  non-diagnosed content-level baseline.
- The uniform baseline's gain (−15.7 pts) comes almost entirely from ME (Sec. 4.5).

**Figure 1** (`figures/fig_ka_me_drift.png`): grouped bars of M1 for the three methods,
pair-clustered 95% CIs.

## 4.5 Per-class analysis

| Subset | n | Native | LSDA uniform | Ours (routed) |
|---|---:|---:|---:|---:|
| ME | 60 | 100.0% | 33.3% | 33.3% |
| KA | 190 | 68.9% | 65.3% | **53.7%** |

- **ME**: local re-diffusion repairs **2/3** of cases; knowledge phrases add nothing (as expected).
- **KA**: without knowledge the content-level fix is nearly inert (68.9 → 65.3); with
  injected attribute phrases it drops by **15.2 points** to 53.7%. The residual is
  *true* knowledge absence, unreachable by any inference-time intervention.

**Figure 2** (`figures/fig_ka_me_perclass.png`): per-class drift bars.

## 4.6 Objective metrics are insufficient for cultural binding (secondary analysis)

| Method | VLM (M1) | CLIP-I | CLIP-T | DINOv2 |
|---|---:|---:|---:|---:|
| Native SS | 63.7% | 75.6% | 75.3% | 69.0% |
| LSDA uniform | 47.8% | 45.2% | 47.2% | 42.5% |
| Ours (routed) | **40.5%** | 49.2% | 49.5% | 45.8% |

Agreement with the VLM judge is only **52–62%**, and the CLIP/DINO metrics **fail to
recover the routed gain** (they rank ours *worse* than uniform). A/B margins are tiny
(CLIP-I ≈ 0.05, CLIP-T ≈ 0.03, DINOv2 ≈ 0.15 with a left/right bias). This matches prior
findings that general-purpose encoders are unreliable for fine-grained cultural
attributes, and motivates the strict VLM protocol used here.

**Figure 3** (`figures/fig_metrics_compare.png`): VLM vs. objective metrics.

## 4.7 Design ablations (LSDA region strategy)

On a controlled single case (pair "Chinese blue-and-white × Italian maiolica", seed 42,
1024) we ablate the region/ownership design:

| Variant | Region | Boundary artifact | Background/shadow |
|---|---|---|---|
| clean v1 | tight SAM mask, hard one-hot | **white seam / 16px staircase** at the object edge | preserved |
| v1.1 | tight mask + soft (area) boundary | seam softened | preserved |
| v1.2 | v1.1 + 8 px dilation | seam removed | preserved |
| **v1.4 (ours)** | **SAM bbox → padded rectangle** | **no seam** | rewritten inside rectangle |

**Figure 4** (`figures/sweep_dilate.png`, `figures/cmp_rect.png`): region-strategy sweep.

## 4.8 Qualitative results

**Figure 5** (`figures/demo_p1_4panel_v14p16.png`): Standalone A | Standalone B | Native SS | LSDA.
The right entity is recovered from Chinese blue-and-white back to Italian maiolica while the
left entity and the scene layout are preserved.

---

# 附：待办与注意事项（中文）

1. **TABLE 1 的 Attention binding 行**：`binding/` 正在 GPU0/3 生成（300 张），完成后用同一 VLM 双裁评分并填入。
2. **Figure 1/2/3** 已生成；Figure 2（per-class）需新画（脚本见下）。
3. **SOTA 说明**：DreamRenderer 需要 FLUX（本环境无权重）、Re-Imagen/RAVEL 需训练/图谱，
   无法同数据重跑；正文以"路由级基线（我们按 DreamRenderer 的注意力绑定重实现在 SD3.5）"
   作同数据集 SOTA 对照，其余在 Related Work 引用其原文数字并注明不可直接比较。
4. **CI**：已用 pair-clustered bootstrap；Wilson 版留作附注。
5. **结构分下降**（0.938→0.72/0.75）必须作为代价报告，不能只报 drift。
