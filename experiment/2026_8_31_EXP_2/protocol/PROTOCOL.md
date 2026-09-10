# 2026_8_31_EXP_2：LSDA 全自动统一评估（协议）

> 本文档是本次评估的协议记录，供后续追溯。数据来源为 EXP_1（2026_8_25_EXP_1），
> 本实验只做**评价**，不生成新图像。

## 1. 目标

用**全自动、本地可跑**的统一指标重新评估 LSDA clean v1（对比 native SS），
不依赖 VLM API；并利用 EXP_1 已有的 Qwen/Gemini 盲评做交叉验证。

## 2. 数据

- 图像（均为 2026_8_25_EXP_1 冻结 seed 产物）：
  - `data/SS/2026_8_25_EXP_1/`：native SS 900 张（sample_0001..0900.jpg）
  - `data/LSDA/2026_8_25_EXP_1/`：LSDA clean v1 900 张（同名 sample）
  - `data/Standalone_A|B/2026_8_25_EXP_1/`：参照图各 900 张
- 索引：`experiment/2026_8_31_EXP_2/manifests/eval_index.json`
  （由 `binary_vqa_v2/key/blind_map.json` 派生，1800 行，0 缺失）
- 已有盲评：Qwen 1800/1800、Gemini 1800/1800（`binary_vqa_v2/ratings/`）
- **已知数据说明**：实测图像分辨率与 manifest 声明（1024×1024）不符——
  SS/Standalone 为 768×726，LSDA 为 512×470。指标侧统一缩放不受影响，
  但报告应注明该分辨率差异。

## 3. 指标与判定

每个候选图（SS 或 LSDA）的左右实体分别与 A-alone、B-alone 参照比较：

1. **siglip_cp**：MaSC 式 masked-maxcos——SigLIP2 patch token 间
   `mean_{i∈ref} max_{j∈cand} cos(r_i, g_j)`，ref 取参照图前景（SAM mask），
   cand 取候选图实体区域（SAM mask）。
2. **clip_i**：CLIP 图像嵌入余弦（实体区域 crop vs 参照整图）。
3. **clip_t**：CLIP 区域 crop 与实体 Short prompt 文本的余弦。
4. **dino**：DINOv2 [CLS] 嵌入余弦（区域 crop vs 参照整图）。
5. **blip**（辅助）：BLIP-VQA 对区域 crop 问"Is this {实体短提示}?"，
   取 yes 判定。

单侧归属：A vs B 取分数更大者。正确绑定 = 左→A 且右→B。
自动共识 = 四指标多数投票（blip 若可用则加入）。

masks：SAM vit-base 点提示（左 0.28/右 0.72 处 3×3 点阵），
重叠仲裁为 left 独占，背景为补集；输出 `segmentation/masks_<cond>/<sample>_mask.npz` + QC PNG。

## 4. 统计口径（与 EXP_1 一致）

- 最小单元 = pair × seed × 图；逐 seed 计分，不合并。
- drift rate（每条件）+ Wilson 95% CI
- 同图配对转移：restored / worsened / persistent_failure / both_correct
- cluster bootstrap（按 100 个文化 pair 重采样，20000 次）漂移率差值 95% CI
- 配对方向检验：exact McNemar（restored vs worsened）
- 与已有盲评：Cohen's κ（自动共识 vs Qwen / Gemini / 双 VLM 均判失败）

## 5. 纪律

- 密钥不落盘（本实验不使用任何 API key）。
- 原始图像只读不改；派生数据写入本实验目录。
- 失败样本保留并计入统计，不删除。
- 双 VLM 共识终版（both-fail=fail）由已有评分计算，已列入本报告。
