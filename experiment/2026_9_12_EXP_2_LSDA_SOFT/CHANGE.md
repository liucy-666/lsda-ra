# LSDA v1.1 — Soft Boundary 变更记录

> 日期：2026-09-12
> 基线：LSDA clean v1（`code/lsda/lsda_pipeline.py`，md5 `ff9d7ab945beaeb7d15f11b3c7f485bc`）
> 新版本：`code/lsda/lsda_pipeline_v11_soft.py`
> 实验目录：`experiment/2026_9_12_EXP_2_LSDA_SOFT/`

## 1. 动机

clean v1 将图像空间二值 owner mask 以 **nearest 下采样**投到 64×64 latent 网格（1/16 分辨率），
并强制严格 one-hot。细长/斜向的对象边缘会被"吸附"到某一 latent cell，
导致部分对象边缘像素落入背景补集 → 显露**冻结原生 SS 的白色影棚背景**（demo P1 右瓶左下白缝），
并在边缘形成 16px 阶梯状锯齿。

## 2. 变更（唯一函数）

`image_masks_to_latent`：

- 旧：`F.interpolate(..., mode="nearest")` + 冲突按质心仲裁 + 严格 one-hot。
- 新：`F.interpolate(..., mode="area")` 得到**分数覆盖率**；若跨实体和 >1 则按比例归一；
  `background = clamp(1 - Σowner, 0, 1)`。
- 结果：`denoise_specialists` 中的合成 `next = background ⊙ native + Σ owner_i ⊙ cand_i`
  退化为**凸加权混合**，边缘 cell 由实体专家与背景共同承担。

## 3. 影响与代价

- 移除边缘白缝与阶梯锯齿（见 `figures/cmp_v1_vs_v11*.png`、`zoom_fine.png`）。
- v1 vs v1.1 输出差异：`mean|Δ|=7.06`，`10.25%` 像素通道差 >20。
- 代价：**不再是严格 one-hot**。`outside_write_rms`、`background_native_match_rms`
  审计值不再为 0（属设计预期，非 bug）；`clean_and_partition` 的图像空间 one-hot 检查保持不变。

## 4. 复现

```
config: experiment/2026_9_12_EXP_2_LSDA_SOFT/manifests/demo_p1_s42_lsda_v11soft.json
run:    lsda_pipeline_v11_soft.py --config ... --helpers-dir .../helpers --output-root ...
run_dir: .../lsda_runs/runs/demo_p1_bluewhite_maiolica_s42_lsda_v11soft/
```

## 5. 后续候选

- 若边缘仍有轻微偏亮：对 soft owner 做 1 latent cell 膨胀，或对 background 端做补偿。
- 需固定这一版本命名，不得覆盖 clean v1。

## 6. v1.2 — soft + dilation（2026-09-12）

基线：v1.1（`lsda_pipeline_v11_soft.py`）。新版本：`lsda_pipeline_v12_soft_dilate.py`。

- `image_masks_to_latent` 在 area 池化前，对图像空间 owner mask 做
  `ImageFilter.MaxFilter(2k+1)` 膨胀（`DILATE_PX`，env `LSDA_DILATE_PX`，默认 24）。
- 目的：让实体专家拥有"原生轮廓带"，从而**填满/擦除**旧轮廓（用户提出的"不允许无主像素"，
  实际含义是让实体专家覆盖形状不匹配的边缘）。
- 结论（P1/s42，见 `figures/sweep_dilate*.png`）：
  - 膨胀确实消除了边缘白缝；
  - 但膨胀越大，"专家重写区域"越大、**生成轨迹偏移越大**，外观漂移越明显
    （`mean|Δ|`：v11→v12d24 = 18.94；目视 8 < 12 < 24 的漂移）。
  - **dilate=8 在"填满边缘"与"保持外观"之间较平衡**。
- 注意：这不是对"旧轮廓完全消除/填满"的字面强制，而是用膨胀近似；真正的形状提交需迭代重分割。

## 7. v1.3 — no-SAM 试验（2026-09-12，结论：失败）

代码：`lsda_pipeline_v13_nosam.py`（`--sam-model-dir` 变可选；用 `foreground_split_masks` 替代 SAM）。

做法：边界中值估背景色 → 颜色距离阈值前景 → 按前景 x 中位数垂直切分 → 种子连通域 + 填洞。

结果（P1/s42，见 `figures/cmp_sam_vs_nosam*.png`、`cmp_masks_sam_vs_nosam.png`）：

- owner 面积各 **0.461**（应约 0.22），背景仅 **0.078** → mask 退化成**左右半平面**（bbox 覆盖整个半边）。
- LSDA 输出**崩坏**：专家重绘了半张图，背景被改写、对象位移。
- 调 `LSDA_FG_THRESHOLD` 0.18→0.45 **无改善**（前景占比 0.42→0.35，仍无法分离）。

根因：这些文化器物**本身是白/浅色**（瓷），与明亮影棚背景颜色接近，**颜色分割无法区分**；
SAM 的强项是**物体性(objectness)**，不是颜色。因此"不要 SAM"实际意味着"换一个**物体感知**的定位器"
（检测器 / 指代分割 / 人工 box / GroundTruth 布局），而不是简单的颜色前景。

结论：**SAM 在当前设置下不可被朴素颜色分割替代。**

## 8. v1.4 — SAM 仅作定位器 + 矩形区域重生成（2026-09-12，结论：可行）

代码：`lsda_pipeline_v14_rect.py`（基于 v1.2；新增 `rectangularize_owners`，env `LSDA_RECT_PAD`）。

做法：SAM 仍用于定位，但**只取每个 owner mask 的外接矩形**（可 pad），
专家在**整个矩形内**重生成（含矩形内背景）；矩形外仍由背景/native 负责。
配合 v1.1 的软边界，矩形边界落在平滑背景里。

结果（P1/s42，见 `figures/cmp_rect*.png`）：

- **pad=0（紧外接矩形）**：效果好，与 v1.2 紧 mask 相当；对象边缘不再被硬切，无明显白缝。
- **pad=32**：外观漂移明显——专家只接受实体 prompt，矩形内的背景被一并重写，pad 越大漂移越大。
- mask 形状：由 SAM 紧轮廓 → **规整矩形**（`cmp_rect_masks.png`）。

权衡：矩形越大 → 对象边缘越自由（无形状不匹配）→ 但背景被重写越多、漂移越大。
建议 **小 pad（0–16）**。

意义：SAM 退化为"只给框"的定位器后，**mask 形状不匹配问题从根上被绕开**（边界不再压在器物轮廓上）。

## 9. 基座决策（2026-09-12）

对比 v1.2（紧 mask + 软边界 + dilate8）vs v1.4（矩形）：

| 维度 | v1.2 | v1.4 矩形 |
|---|---|---|
| 对象边缘白缝 | 无 | 无 |
| 地面接触投影/背景光照 | **保留**（阴影区 250.5 ≈ native 249.7）| **丢失**（246.1，矩形底边平切）|
| 区域形状 | SAM 紧轮廓 | 规整矩形 |

**决定（2026-09-12，最终）：采用 v1.4（矩形，pad=16）作为后续全量 KA+ME 实验的 LSDA 基座。**
- 理由：用户审图后认为 v1.4 p16 的**外观**最合预期；矩形区域同时消除对象边缘白缝，且 SAM 退化为纯定位器、管线更简单。
- 明确接受的代价：矩形区域会**重写器物周围的地面/背景（含底部接触投影）**，背景光照不再等同原生；此点如实记录为方法代价。
- 代码：`code/lsda/lsda_pipeline_v14_rect.py`，运行参数 `LSDA_RECT_PAD=16`、`LSDA_DILATE_PX=0`。
- 样例：`figures/demo_p1_4panel_v14p16.png`；单图 `data/LSDA/2026_9_12_EXP_1_DEMO_P1/lsda_v14rect_p16.png`。
- v1.2（紧 mask，保留背景光照）保留为**消融/对照**版本；"矩形外观 + 保留原生投影"的混合版列为未来候选。
