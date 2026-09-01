# Agentic LSDA：序贯工具调用数据管线

本目录是独立于 `code/lsda` clean v1 的探索版本。它不修改 v1 的 step-0 定义，目标是构造
“同一原生轨迹、不同 LSDA 调用时间与调用对象”的反事实数据，为后续 Oracle 蒸馏和 RL
策略训练提供可追溯环境轨迹。

## 当前完成边界

- `build_counterfactual_manifest.py`：从冻结的 900-task manifest 和 Qwen 配对转移表中，
  按 `restored / persistent_failure / worsened / both_correct` 分层抽取 pilot，并展开 WAIT 与
  `(step, target)` 干预动作。
- `schema.py` / `validate_manifest.py`：冻结 replay schema 并检查动作、时刻、Prompt 和 ID。
- `timed_lsda.py`：从原生 checkpoint 分支；被选实体由 Short-Prompt 局部专家接管，背景与
  未选实体继续使用同 seed 原生 SS 轨迹。
- `rl_env.py`：定义 `WAIT / INVOKE / ACCEPT / ROLLBACK / TERMINATE` 状态机、工具预算、
  后干预观察窗口，以及绑定收益受结构下限约束的可审计奖励接口。

当前尚未声称可训练 VLM/RL：仓库还没有 MM-DiT 的 MLP、attention 和 residual-flow hook，
也没有反事实候选图的绑定/结构评分。第一阶段先生成 timing-oracle 数据，再接入内部观测和奖励。

## 动作语义

```text
WAIT:
  完整轨迹使用 frozen native SS。

INVOKE(step=t, target=A|B|AB):
  t 之前使用 frozen native SS；
  t 起被选 owner 使用对应实体 Short Prompt 的局部专家；
  背景和未选 owner 继续使用 frozen native SS states。
```

这是一种反事实单次干预 primitive。后续多轮 Agent 环境会在此基础上加入 checkpoint、
干预后观察、ACCEPT/ROLLBACK 和第二次 INVOKE。`rl_env.py` 已冻结该闭环接口，但真实
SD3.5 backend 仍需通过 GPU smoke test；不要把未运行的接口描述成已训练 Agent。

## Pilot manifest

```powershell
python code/agentic_lsda/build_counterfactual_manifest.py `
  --source-manifest experiment/2026_8_25_EXP_1/cultural100_records/experiment_4500/lsda_manifest.json `
  --transitions-csv experiment/2026_8_25_EXP_1/cultural100_records/experiment_4500/binary_vqa_v2/source_data/qwen_paired_transitions.csv `
  --mask-npz-root experiment/2026_8_31_EXP_2/segmentation/masks_native_SS `
  --output experiment/2026_9_1_EXP_1/manifests/counterfactual_pilot.json

python code/agentic_lsda/validate_manifest.py `
  experiment/2026_9_1_EXP_1/manifests/counterfactual_pilot.json
```

默认 pilot 为 12 个原始样本，7 个候选时刻，三个 target，加每个样本一个 WAIT，共 264 个
replay task。正式生成前应先用 2 个样本 × 3 个时刻完成 GPU smoke test。

远程 GPU 生成入口：

```bash
python code/agentic_lsda/collect_counterfactuals.py \
  --manifest experiment/2026_9_1_EXP_1/manifests/counterfactual_smoke.json \
  --model-dir /ABS/PATH/stable-diffusion-3.5-large \
  --native-images-root data/SS/2026_8_25_EXP_1 \
  --mask-npz-root experiment/2026_9_1_EXP_1/segmentation/source_native_masks \
  --image-output-dir data/Agentic_LSDA/2026_9_1_EXP_1 \
  --record-output-dir experiment/2026_9_1_EXP_1
```

路径必须由运行者显式传入；脚本不包含 `/science/...`、`AAA_Experiment` 或其他机器专用路径。

## 数据落盘纪律

- 候选图像：`data/Agentic_LSDA/<experiment-id>/...`
- manifest、sidecar、指标和日志：`experiment/<experiment-id>/...`
- 不覆盖 EXP_1、LSDA clean v1 图像或评分。
- 机制 hook 未接入的字段必须明确为空，禁止用最终标签冒充在线观测。
