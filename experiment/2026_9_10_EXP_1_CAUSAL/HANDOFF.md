# HANDOFF：2026_9_10_EXP_1_CAUSAL 机制分析（Agent 交接）

> 更新：2026-09-11
> 权威规范：`warning.md`；LSDA 方法：`code/lsda/README.md`
> 本实验目录：`experiment/2026_9_10_EXP_1_CAUSAL/`；图像：`data/Causal/2026_9_10_EXP_1/`

---

## 1. 实验目标

在 **MM-DiT（SD3.5-large）** 的文化实体组合生成中，回答：

> 属性漂移（如右瓶意式彩陶被画成中式青花）的差异，**主要来自注意力路由（W）还是内容（V）**？

贡献定位：**analysis-level 的新视角 + 测量框架**（非新模型）。

---

## 2. 最终结论（以最新、方向感知口径为准）

> **内容（Value）与路由（Attention）共同作用；但内容携带了朝向正确答案的主要修正方向——约 3.6 倍于路由。**

| 指标（5 对，N=33 seeds，1024×1024） | 值 |
|---|---|
| `proj_V`（换内容补上的差距） | **0.82 ± 0.76** |
| `proj_W`（换路由补上的差距） | **0.23 ± 0.85** |
| 比值 | **3.56** |

逐对：003=0.42/0.98、013=−0.55/0.12（异常）、015=0.36/0.91、046=0.74/1.20、084=0.36/1.03。

终局图 2×2（pair 1, N=9, 512，单点 L36/t26）：

| 臂 | 恢复量 | 翻转 | sign p |
|---|---|---|---|
| w_fix | +0.083 | 5/9 | 0.50 |
| v_fix | +0.089 | 6/9 | 0.254 |
| **both_fix** | **+0.172** | **8/9** | **0.0195** |

→ 单独修任一项均不显著；**同时修才显著**。

---

## 3. ⚠️ 撤回记录（务必阅读）

| 项 | 状态 | 原因 |
|---|---|---|
| ~~h_fix（残差流替换）能修复~~ | ❌ **撤回** | `patch_h_in` **维度 bug**：在 `[seq,d]` 上用 `dim=1`（替换特征列而非 token）。修正为 `dim=0` 后：最后一层不修复、多层崩溃 |
| ~~"变化量"口径的 65/35~~ | ❌ **作废** | 旧指标 `‖O_mixed−O_fixed‖` 只测"变了多少"，乱变也得高分；已改为方向感知 `proj`（82/23） |
| 窗口版 2×2（L33-36×t24-27） | ⚠️ 回收 | 效应弱（+0.036），被单点版取代 |
| 文化轴投影 | ⚠️ 弱证据 | 跨对不一致（pair_046 反向），不作主证据 |

**教训**：① 张量维度必须核对形状（512 下越界不报错会静默出错）；② VLM 响应可能被截断（思考型模型需 `reasoning_effort=none`）；③ "成功了"要先用最保守方式复验。

---

## 4. 文件地图

### 4.1 保留（核心，最新且正确）

**图（`figures/`）**
| 文件 | 内容 |
|---|---|
| `fig_probe_direction_aware.png` | **主结果**：5 对 proj_W vs proj_V + 逐 seed 散点 |
| `fig2_pair1_recovery.png` / `fig3_pair1_scatter.png` | 2×2 终局验证（单点） |
| `fig4_pair1_responsibility.png` | route/value 责任条（单点） |
| `fig_schematic_block.png` | 框架示意 |
| `fig_deepdive_leakage.png` | 泄漏复现 |
| `fig_deepdive_hfix_corrected.png` | h_fix 负面结果 |
| `axis_pair_*.png` | 文化轴（弱，辅助） |
| `demo_hfix_last/fixed/win_1009.png`、`demo_singlepoint_pair1.png`、`demo_strong_1009.png` | 定性对照 |

**记录**
| 路径 | 内容 |
|---|---|
| `deepdive/probe_v2/` | **方向感知探针（主证据）** |
| `deepdive/axis/` + `axis_analysis/` | 文化轴投影（弱） |
| `deepdive/deepdive_analysis.json` | 新裁判泄漏判定 |
| `deepdive/manifest_deepdive.jsonl` | 深挖评分清单 |
| `e2_analysis/` + `scores_e2_pair1.jsonl` + `manifest_e2_pair1.jsonl` | **单点** 2×2 |
| `manifest_e0_pair1.jsonl` + `scores_e0_pair1.jsonl` | E0 纳入门槛 |
| `e1_pair1_seed1000.json` | E1 复现（分解恒等式 2e-4） |
| `EXPERIMENTS_DRAFT.md` / `METHOD_DRAFT.md` / `REPORT.md` | 文档 |

**图像（`data/Causal/2026_9_10_EXP_1/`）**
| 目录 | 内容 |
|---|---|
| `deepdive1024/e0/`（111） | 5 对 native SS + standalone @1024 |
| `deepdive1024/arms/`（66） | v_fix @1024 |
| `e2strong/`（4） | v_fix 全层（无修复） |
| `e2hfix_fixed/last/win/`（各 2） | 修正版 h_fix |
| `e2/`（72）、`e0/`（120） | pair-1 单点 2×2 与对照 |

### 4.2 回收（`_recycle/`，勿用于正式结果）
- `records/probe_old_metric/`（旧口径探针）
- `records/e2_internal_old_metric/`（pair-1 旧口径）
- `records/e2win_analysis_window/`（窗口 2×2）
- `figures/fig_internal_share.png`、`fig_main_result.png`、`fig_deepdive_probe_5pairs.png`、`demo_hfix_1009.png`、窗口版 fig2/3/4
- `data/e2hfix_buggy/`、`data/e2win_window/`
- `temp/`（日志、审计、review 中间件）

---

## 5. 可复现步骤

**环境**：服务器 `ssh -p 36111 wx@s3.v100.vip`；python `/science/wx/pry/.venv/bin/python`；代码 `/science/wx/pry/EXP1/code`；模型 `/science/wx/pry/models/stable-diffusion-3.5-large`。

1. **重建历史失败清单**：`python code/mmdit_causal/reconstruct_failures.py`（→ `deepdive/deepdive_manifest.json`）
2. **定向重生成 @1024**：`gen_deepdive.py --jobs deepdive_jobs_*.json --out .../deepdive1024/e0`
3. **新裁判评分**：`score_images.py --manifest ... --out ... --rater GPT54,GEMINI35`（`gemini-3.5-flash:stable`；**必须加 `reasoning_effort=none`**）
4. **泄漏判定**：`analyze_deepdive.py`（→ `deepdive_analysis.json`）
5. **方向感知探针**：`run_deepdive_probe_v2.py --pair N --seeds ...`（→ `deepdive/probe_v2/`）
6. **出图**：`make_fig_probe_v2.py`（Fig 1）、`analyze_e2.py --scores scores_e2_pair1.jsonl`（Fig 2）、`make_deepdive_figs.py`（Fig 4/5）
7. **美学审计**：`audit_figures.py`（gpt-5.6-sol）

---

## 6. 开放问题与下一步

1. **pair_013 异常**（proj_W=−0.55、proj_V=0.12）：需单独诊断（是否轴/ROI 问题）。
2. **纳入门槛样本偏少**：5 对 45 seed 中仅 7 个满足 A✓B✓SS✗；结论需在门槛内样本上单独报告。
3. **窗口选择**：沿用历史窗口（L33-36×t24-27），未做窗口扫描；建议补扫描验证普适性。
4. **Δ 的假设**：以 donor（B-only 分支）为"正确"参照，是相对定义；需在论文中写明。
5. **2×2 终局图分辨率**：pair 1 @512，而探针 @1024；建议统一或在论文中说明。
6. **文化轴**：当前弱，需改进 ROI 对齐/轴构建后再考虑入正文。

---

## 7. 关键脚本（`code/mmdit_causal/`）

| 脚本 | 作用 |
|---|---|
| `reconstruct_failures.py` | 重建历史双裁严格失败清单 |
| `gen_deepdive.py` | 定向重生成（native SS + standalone） |
| `run_deepdive_probe_v2.py` | **方向感知探针**（主实验） |
| `run_deepdive_arm.py` | v_fix / h_fix 干预 |
| `run_axis.py` | 文化轴投影 |
| `score_images.py` | 双 VLM 评分（GPT-5.4 + Gemini-3.5-flash） |
| `analyze_deepdive.py` | 泄漏判定 |
| `make_fig_probe_v2.py` / `analyze_e2.py` / `make_deepdive_figs.py` | 出图 |
| `audit_figures.py` | gpt-5.6-sol 美学审计 |
| `mmdit_lib.py` | Controller/安装/干预核心（**h_fix 维度已修正为 dim=0**） |
