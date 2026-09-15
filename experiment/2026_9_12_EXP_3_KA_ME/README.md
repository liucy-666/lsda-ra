# 2026_9_12_EXP_3_KA_ME — 统一知识注入 + 局部重扩散（Result 2 受控实验）

## 目的
在受控 100 文化对 × seeds 1011/1012/1013 = 300 样本上，验证「统一知识注入 + 局部重扩散（LSDA v1.4 矩形）」相对原生 SS 降低文化错绑；并做 KA/ME/BC 普查与路由级 binding 对照。

## 数据与产物
- 图像：`data/KA_ME/2026_9_12_EXP_3/jpg/`（A/B/SS 900；`lsda_ka` 189、`lsda_meattrs` 60、`lsda_uniform` 189、`lsda_me` 60）、`binding2_jpg/` 300。
- 元数据：`analysis/classification.json`（300 样本 KA/ME/BC）、`figures/`、`ratings/`、`manifests/`。
- 代码：`code/lsda/lsda_pipeline_v14_rect.py`、`code/mmdit_causal/*`、`code/metrics/*`。

## 结论
- 三分类：**KA 190（63.3%）、ME 60（20.0%）、BC 50（16.7%）**；原生失败中 **KA（知识缺失）占 76%**。
- 5 指标 × 3 臂（drift %，越低越好）：

| 指标 | SS | LSDA（统一知识注入） | Binding v2（路由级） |
|---|---:|---:|---:|
| VLM (c1) | 66.7 | **15.4** | — |
| LBP/GLCM | 80.0 | **72.2** | 83.3 |
| Gram | 74.3 | **54.8** | 79.7 |
| k-NN | 69.7 | **48.8** | 77.3 |
| SigLIP | 84.7 | **66.2** | — |

- **LSDA 全部指标改善**；**路由级 binding 在客观指标上比 SS 更差**（方法有害/无效）。
- 分流（KA/ME）vs 统一：只差 2 个样本 → 负责人决定方法收缩为「统一知识注入」，KA/ME 降级为分析。

## Caveats
- `LSDA-attrs` 复用自本实验，规则不一致（KA=只注入失败侧；ME=两侧）；后续 `LSDA-Long` 用统一规则。
- VLM 判据：**c1（rater-strict）为主**，c2/c3 附注；`RESULTS.md`/`PAPER_METHOD_EXP.md` 旧口径数字不用。

## 进度与未来
已完成。未来：全量 1000 对基准（见 `2026_9_14_EXP_5`、`2026_9_14_EXP_4`）；统一 attrs 规则；注入位置对比见 `2026_9_15_EXP_6`。
