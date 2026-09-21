# 2026_9_15_EXP_7_FOOD_PILOT — food 可视化可行性 pilot

## 目的
判断"food（菜肴）"是否适合做文化绑定的大规模可视化 benchmark。只做 3 对 × 4 臂 = 12 图，目视为主。

## 冻结设计
- 3 对（跨文化、食材/形态差异最大化）：
  1. Nigeria egusi × South Korea gimchobap
  2. Brazil bacalhau a bras × Italy caffe con panna
  3. Brazil salpicao × Italy spaghetti carbonara
- 臂：standalone A / standalone B / 原生 SS / LSDA-attr（区域专家 = `short + attrs`）。
- 生成：SD3.5-large，1024，28 steps，CFG 4.5，fp16，**seed 42**。
- 模板：`Neutral studio background: {A} on the left, {B} on the right; both fully visible, separate, and similar in size.`
- 图：`data/Food_Pilot/2026_9_15_EXP_7/{A,B,SS,LSDA_Attr}/`（服务器 `ssh A100` 生成）。

## 脚本
- `code/culture_comp/gen_pairs_census.py`：manifest 驱动的 A/B/SS 生成（命名兼容 `code/lsda/run_census_lsda.py`）。
- `code/lsda/run_census_lsda.py`：LSDA v1.4 矩形（`RECT_PAD=16`），a/b_prompt 用 `short + attrs`。

## 观察点
1. standalone 是否生成所标文化（KA：知识缺失？）。
2. 原生 SS 是否串味/混为一团。
3. LSDA-attr 是否把两实体拉回各自文化。
4. 是否出现矩形截断/灰条伪影。

## 结果（seed 42，目视；未做 VLM 评分）
图：`figures/fig_food_pilot.png`（行=pair，列=A/B/SS/LSDA_Attr）；图源 `data/Food_Pilot/2026_9_15_EXP_7/`。

- **p001 egusi × gimchobap**：A=绿色浓汤（尚可）；**B standalone 失败**（生成成肉卷/面包环，非紫菜包饭）→ KA；SS 两侧都成红汤（缺知识/漂移）；**LSDA-attr 右侧生成了可辨认的 gimbap（紫菜包饭）**——区域专家把缺失知识补了回来；代价：左出现怪异整颗蛋/土豆 + 矩形硬边与灰条。
- **p002 bacalhau à brás × caffè con panna**：A=白色块状（不明确）；B=奶泡咖啡（可）；SS 左白块+右咖啡；LSDA-attr 右侧变成**分层咖啡（更符合 caffè con panna）**，但左出现勺状/横条伪影 + 灰带。
- **p003 salpicão × carbonara**：A=彩色沙拉（好）；B=意面（好）；SS 基本分离；LSDA-attr 两侧都对，但**中间出现竖直棍状 + 灰条伪影**（矩形边界）。

## 结论
1. **food 的 standalone 知识不均衡**（gimchobap 失败、carbonara/caffè 成功）→ 继续印证 KA 问题。
2. **LSDA-attr 能补回缺失知识**（p001 右=gimbap 是亮点），支持"KB → 局部注入"路线。
3. **v1.4 矩形伪影在 food 上非常明显**（灰条、硬边、棍状），正式评测前必须先修 v1.5（mask/羽化提交）。
4. food 绑定判读偏弱（碗状同质）：主基准仍应用"清晰器物+表面纹样"，food 作为**知识缺失/困难层**，并把 gimbap 这类实例作为"局部注入补知识"的定性证据。

## 进度
- [x] 生成 12 图（A/B/SS/LSDA-attr，seed 42，1024）
- [x] 目视 + 报告
- [ ] （可选）v1.5 修复伪影后复测同一 3 对
- [ ] （可选）换更清晰 food 粒度（卷类/糕点/整只烤物）复核

---

## 追加：LSDA v1.5 两种提交掩膜（美学取向，10 对 × seed 1011）

**目的**：消除 v1.4 矩形硬边界的"灰色直缝"；美学质量交由负责人目视评判（无 VLM）。

**两方向**（独立实现于 `code/lsda/lsda_pipeline_v15.py`）：
- **方向1 `v15mask`**：提交掩膜改回 **SAM 紧轮廓**（dilate=24）+ 羽化 12；矩形只用来算 crop。
- **方向2 `v15alpha`**：保留**矩形**（pad=48）但用**羽化 alpha 融合**，让专家重画的矩形内背景与原生背景交叉淡化（不再硬切）。

**配置**：`v14rect`=基线（矩形 pad16、feather 0）；`v15mask`=mask/dilate24/feather12/pad16；`v15alpha`=alpha/pad48/feather16。
**10 对**（受控 100 对中选，美学取向）：001,002,004,008,014,016,022,041,061,081。

**代码/脚本**：`code/lsda/lsda_pipeline_v15.py`、`code/lsda/run_census_lsda_v15.py`、`code/lsda/run_aesth_v15.sh`。
**图**：`figures/fig_aesth_1.png`（p001–014）、`fig_aesth_2.png`（p016–081）、`figures/fig_aesth_seamzoom.png`（中心放大）。
**图源**：`data/Aesth_Pilot/2026_9_15_EXP_7/{SS,v14rect,v15mask,v15alpha}/`（40 张，1024）。

**观察（负责人已评判）**：
1. `v14rect`：沿矩形边界出现硬直边 + 露出的原生影棚背景条 → 原"灰线"。
2. **`v15mask` 胜出（选定方向）**：直边消失，盘/瓶轮廓更完整，美学最好。
3. `v15alpha`（pad48）：矩形内背景被一起重画，出现**矩形背景块/残影**，弃用。
4. **决定（2026-09-15）**：LSDA 提交掩膜采用 **`v15mask`（SAM 轮廓 + dilate24 + feather12，矩形仅算 crop）**。

## 追加问题：物体只有一半（尺寸/构图）
**结论：不是分辨率**。全部生成图均为 1024×1024；"盘子只有一半"来自**原生 SS 的构图**——物体过大、从画框两侧溢出，LSDA（mask 模式）只保留该构图，无法在画框外补内容。

**证据（SS 前景贴边率，128px 前景检测）**：
| pair | 内容 | L | R | B |
|---|---|---:|---:|---:|
| 002 | 盘 | 0.80 | 0.75 | 0.00 |
| 014 | 盘 | 0.82 | 0.84 | 0.00 |
| 016 | 盘 | 0.74 | 0.72 | 0.00 |
| 081 | 地毯 | 0.91 | 0.87 | 0.00 |
| 001/004/008/022/041 | 瓶/罐 | ~0.1 | ~0.2 | 1.0（底部接触，正常）|

→ 盘/地毯类左右贴边 0.7–0.9 = 物体横向被截断。

**修复方向（在 SS prompt 模板层）**：加"whole objects fully inside the frame with generous margin, not cropped, zoomed-out, equal size, centered side-by-side"，并做**构图 QC 选种子**（左右贴边率低才用）。现测试见下。

## 追加：SS 构图修复测试（v15mask）— 结果
对盘类 002/014/016，用 3 套"全景留白"模板重写 `ss_prompt`，跑 v15mask（`figures/fig_fixframe.png`，列=current/T1/T2/T3，每格 [native_ss | lsda]）。

**模板**：
- T1：`... both whole objects fully inside the frame with generous empty space around them, not cropped, centered composition, similar size.`
- T2：`Top-down ... each shown in full with a wide margin, not touching the frame edges ...`
- T3：`... wide, zoomed-out ... lots of empty space around them ... fully visible, not cropped ...`

**贴边率（前景 L/R，越低越好）**：
| pair | current | T1 | T2 | T3 |
|---|---|---|---|---|
| 002 | 0.80/0.75 | **0.18/0.30** | 0.00/0.00 | 0.38/0.61 |
| 014 | 0.82/0.84 | 0.52/0.72 | 0.53/0.67 | 0.64/0.70 |
| 016 | 0.74/0.72 | 0.71/0.65 | 0.65/0.59 | 0.74/0.71 |

**结论**：
1. 分辨率不是问题（全部 1024）；"半个盘子"= **原生 SS 物体贴边/溢出**，LSDA 只能继承。
2. **T1 最好**（p002 贴边 0.8→0.18，输出整只盘子）；但 p014/p016 仍偏高 → 单靠模板不够，需 **构图 QC + 选种子**（生成多个 seed，保留左右贴边率低者）。
3. 最终配方：**T1 模板 + 贴边率选种 + v15mask 提交**。

**进度**：v15mask 已选定；构图修复定为「T1 + 贴边率选种」；待批量重生成 10 对复核。

## 追加：构图修复（T4「小尺寸+大留白」+ 多 seed 选种 + v15mask）— 结果
**模板 T4**：`... shown small and fully inside the frame, each object occupying only a small part of the image with wide empty margins on all sides, lots of negative space, not cropped, equal size, centered side by side.`（注意：>77 token 触发 CLIP 截断，由 T5 承担长文本）
**选种**：每对 5 seeds（1011–1015），按**左右贴边率最低**选（并要求左右半区都有前景）。脚本 `code/culture_comp/select_ss_seed.py`。

| pair | 选中 seed | 贴边 L/R |
|---|---|---|
| 001 | 1015 | 0.20/0.28 |
| 002 | 1014 | 0.00/0.00 |
| 004 | 1011 | 0.16/0.16 |
| 008 | 1013 | 0.07/0.31 |
| 014 | 1013 | 0.00/0.15 |
| 016 | 1013 | 0.07/0.00 |
| 022 | 1014 | 0.52/0.06 |
| 041 | 1011 | 0.22/0.16 |
| 061 | 1011 | 0.00/0.05 |
| 081 | 1015 | 0.00/0.00 |

**结果（`figures/fig_small_1.png`、`fig_small_2.png`；列=old SS / new SS(selected) / new LSDA(v15mask)）**：
- 多数对（001/002/004/008/022/041/061/081）**盘子/瓶完整入框**，v15mask 输出整只器物 → "半个盘子"已解决。
- **残余问题**：p014 选出的大盘横跨画面、第二只盘缺失/合并；p016 两盘重叠成"半盘拼贴"。简单"左右半区有前景"检测区分不出"一个 vs 两个"。
- 改进方向：加**中心背景谷 QC**（中间 20% 列应多为背景，即两物体分开）或按器型定制模板、增加 seed 数。

**脚本**：`gen_pairs_census.py --kinds SS`、`select_ss_seed.py`、`make_selsmall_montage.py`；图源 `data/Aesth_Pilot/2026_9_15_EXP_7_selsmall/`。
**进度**：T4+选种+v15mask 配方成立；待处理 p014/p016 的"单物体/重叠"。
