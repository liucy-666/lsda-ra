# 2026_9_1_EXP_1 — Agentic LSDA 反事实时机 Pilot + SFT 诊断器（支柱 ③）

## 目的
验证 LSDA 最优介入时机是否随样本/实体变化，构建用于 Oracle 蒸馏的第一批反事实轨迹，并训练漂移诊断器。本实验不是 RL 正式训练。

## 设计
- 抽样：按 Qwen transition 分层 12 样本（5 restored / 3 persistent / 2 worsened / 2 both-correct）。
- 候选介入步：0/4/8/12/16/20/24；target：A/B/AB；每样本另存 WAIT 原生轨迹。
- 冻结 RL 环境接口 v0：`code/agentic_lsda/rl_env.py`（Observation / Action{WAIT,INVOKE,ACCEPT,ROLLBACK,TERMINATE}/Pending 规则）。
- 原生 masks 打包于 `segmentation/source_native_masks/`，不依赖旧服务器目录。

## 结论
- SFT 诊断器 v6（Qwen2.5-VL-3B + LoRA）：终局 drift AUC **0.773**、leakage AUC **0.831**。
- checkpoint 级 264 分支（12 样本 × 7 时刻 × A/B/AB）：聚合概率随介入时机弱变（native 0.495 / t=0 0.460 / t=24 0.487），但**逐样本区分度弱**（range ≤ 0.13）。
- 时机曲线见 `h2_out/`；评分为 `scores_pilot*.jsonl`、`pilot_drift_scores_v6*.json`。

## 进度
pilot + 诊断器已完成；**策略训练（BC/DPO/RL）未开始**。

## 未来方向
1. reward 源：偏好型 reward（分支两两比较）或诊断器暖启动 + 偏好接棒。
2. 扩样本至 50–80 确认逐样本时机差异。
3. 结构评分（C2）补齐后才能产出正式 Oracle 标签。
