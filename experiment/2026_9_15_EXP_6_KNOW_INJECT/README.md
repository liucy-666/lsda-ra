# 2026_9_15_EXP_6_KNOW_INJECT — 知识注入位置实验（LL vs LSDA）

## 目的
在**同一份知识**下对比注入位置：**LL（全局长提示）** vs **LSDA-Long（统一长文本局部注入）** vs **LSDA-attrs（旧蒸馏知识）**。核心问题：给定相同文化知识，区域局部注入是否优于全局提示工程？

## 数据与臂（100 对 × seeds 1011/1012/1013）
| 臂 | 全局 prompt | 区域专家 prompt | 知识 | n |
|---|---|---|---:|---:|
| SS | short A/B | - | 无 | 300 |
| LL | 组合 Long (A+B) | - | long | 300 |
| LSDA-Long | short SS（原生轨迹冻结） | long_A / long_B | long | 249 |
| LSDA-attrs | short SS | short+attrs_A / short+attrs_B | attrs（复用） | 249 |

- 图像：`data/KNOW_INJECT/2026_9_15_EXP_6/{LL,LSDA_Long}/`；评分/分析在本目录。
- LSDA：`code/lsda/lsda_pipeline_v14_rect.py`（矩形 pad=16），仅换区域专家，无参考图。

## 结论
- 修复率（原生 SS 失败中判正确）：**LSDA-Long 83.3%**、LSDA-attrs 79.4%、**LL 51.5%**（c1）。
  - Bootstrap：LSDA-Long − LL = **+31.0 pts [23.2, 38.7]**；LSDA-Long − LSDA-attrs = +3.5 (n.s.)。
- 误伤原生正确样本：LSDA 5–10%，**LL 34%**。
- 客观 4 指标上 **LL 比 SS 还差**（Gram 79.3 vs 74.3；k-NN 72.7 vs 69.7；SigLIP 88.0 vs 84.7）。
- **结论：同知识下注入位置决定成败；全局提示工程无效甚至有害。**

## Caveats（必须随结果报告）
1. LSDA-attrs 沿用 exp3 旧资产，规则不一致（KA 只失败侧 / ME 两侧）；LSDA-Long 统一。
2. attrs 是 long 的有损蒸馏 → `LL vs LSDA-attrs` 为「处理+注入」联合效应；`LL vs LSDA-Long` 才是纯位置效应。
3. `p073_s1012` 缺失（empty-latent）→ LSDA 臂 n=249，如实计入。
4. LL 超 77 token 触发 CLIP 截断（T5 承担长文本），正常。
5. 美学审计改用 gemini-3.5-flash；结果全 overlap=false、aes 7–9。
6. **矩形硬边伪影**：LSDA v1.4 rect 会在矩形边界直切物体、矩形间缝隙保留原生背景形成灰条（非分辨率问题；管线 `SIZE=1024`，本地 768 为评测降采样）。

## 进度与未来
分析完成。未来：① 将本结果并入论文 Method/Experiment；② 对 LSDA 图像做**伪影审计**；③ 新版本 v1.5（mask/羽化提交）修复矩形截断；④ 大规模 LSDA-Attrs（1000 对 × 1 seed，QC ≥1024×1024）。
