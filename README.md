# MMDIT — 文化绑定漂移：度量、干预与 Agent（项目总览）

研究 SD3.5 Large 在多文化实体组合生成中的**属性泄漏 / 错误绑定**（如右瓶被左瓶文化污染）。

## 三支柱与当前结论

**① 发现与解释** `[data-supported]`
- **Result 1（机制）**：把病灶从注意力路由 W 重新定位到内容 V。方向感知探针 `proj_V=0.82` vs `proj_W=0.23`（5 对 × 33 seeds，≈3.6×）；终局 2×2 单修任一因素不显著，**同修才稳定翻转**（8/9, p=0.02）。见 `experiment/2026_9_10_EXP_1_CAUSAL/`。
- 普查（受控 100 对）：原生失败中 **KA（知识缺失）占 76%**；分类 KA 190 / ME 60 / BC 50。见 `experiment/2026_9_12_EXP_3_KA_ME/`。

**② LSDA 干预** `[data-supported]`
- LSDA clean v1（step-0 区域专家、严格 one-hot；基座为 v1.4 矩形 mask，`RECT_PAD=16`，见 `code/lsda/README.md`）。
- 900 任务严格双 VLM 交集：原生失败 25.67% → LSDA 3.11%，**原生失败恢复 90.91%**，损伤 1.05%。
- 受控 100 对（VLM c1）：66.7% → 15.4%；4 客观指标全面改善；路由级 binding 基线反而更差。
- **EXP_6 注入位置**：同一知识下，修复率 **LSDA-Long 83.3% ≈ LSDA-attrs 79.4% > LL 51.5%**（+31.0 pts，CI 不含 0）；LL 误伤 34%、客观指标比 SS 还差 → **位置决定成败，全局提示工程无效甚至有害**。见 `experiment/2026_9_15_EXP_6_KNOW_INJECT/`。

**③ Agent Planner** `[system design]`
- 环境接口（`code/agentic_lsda/rl_env.py`）、264 反事实分支、诊断器 v6（drift AUC 0.773 / leakage AUC 0.831）就绪；**策略训练未开始**。见 `experiment/2026_9_1_EXP_1/`。

## 当前工作重心
1. **可视化文化实体选取 + 知识库重构**：复用 CUBE/TU 并做 Wikidata 反查，新增结构化视觉属性（material/palette/motif/technique/form）供 LSDA-Attrs 注入（规划 EXP_7）。
2. **大规模 LSDA-Attrs 指标**：1000 对 × 1 seed(42)，五指标（双 VLM c1/c2/c3 + LBP/GLCM、Gram、k-NN、SigLIP）；分辨率 QC：任何输出 < 1024×1024 视为不合格（规划 EXP_8）。
3. 已知待修：v1.4 矩形 mask 会在矩形边界**直切物体、矩形间缝隙保留原生背景形成灰条**（非分辨率问题，管线 `SIZE=1024`）。

## 目录
- `code/`：源代码（每子目录一份 README）。
- `data/`：仅图像（按方法/实验分层，已 gitignore）。
- `experiment/`：协议、manifest、评分、分析、图表（每个实验目录**一份 README.md**）。
- `tmp/`：回收站（`tmp/recycle_2026_09_15/`，含 `MOVES.tsv` 可追溯/可还原；已 gitignore，不入 GitHub）。

## 实验索引
| 目录 | 内容 |
|---|---|
| `2026_8_25_EXP_1` | LSDA clean v1 全量 900 任务基线（Result 2 主证据） |
| `2026_9_1_EXP_1` | Agent 反事实时机 Pilot + SFT 诊断器（支柱 ③） |
| `2026_9_10_EXP_1_CAUSAL` | Result 1：内容 V 主导路由 W |
| `2026_9_12_EXP_2_LSDA_SOFT` | LSDA 边界/区域版本消融（v1.1→v1.4） |
| `2026_9_12_EXP_3_KA_ME` | 受控 100 对：统一知识注入 + 局部重扩散 |
| `2026_9_14_EXP_4_CULTURE_KB` | 外置文化知识库（3546 条） |
| `2026_9_14_EXP_5_CULTURE_COMP` | 文化组合 benchmark（采样 + pilot） |
| `2026_9_15_EXP_6_KNOW_INJECT` | 知识注入位置（LL vs LSDA-Long/attrs） |

## 纪律
一切以 `warning.md` 为准：原始数据只增不改、失败样本不隐藏、判据全流程一致、`TEST_API_KEY` 不落盘、VLM 评分 `reasoning_effort=none`、SSH/API 三振即停、长任务必配看门狗。

## Plan 9/13 数据路径
`code/kb`、`code/benchmark`、`code/metrics/structure.py`、`code/mmdit_causal/value_projection.py` 收录了 plan 9/13 的**免模型**工具：归一化 CUBE/TU、构建可审计 Wikidata KB、生成确定性跨文化 prompt、报告结构与 Value 方向代理，均不需要本地模型权重。本地检查与服务器命令见 `docs/plan_9_13_implementation.md`。
