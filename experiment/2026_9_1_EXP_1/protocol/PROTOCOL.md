# 2026_9_1_EXP_1：Agentic LSDA 反事实时机 Pilot

## 目的

验证 LSDA 的最优介入时机是否随样本和实体变化，并构建用于 Oracle 蒸馏的第一批反事实轨迹。
本实验不是 RL 正式训练，也不声称现有内部指标已经能够在线预测漂移。

## 冻结来源

- Prompt / seed：`2026_8_25_EXP_1` 的 900-task `lsda_manifest.json`。
- 初始分层标签：同实验 Qwen 配对转移表，仅用于 pilot 抽样，不作为最终奖励真值。
- 模型：Stable Diffusion 3.5 Large。
- 调度：FlowMatchEulerDiscreteScheduler，28 steps，CFG 4.5。
- LSDA v1 保持不变；本实验使用独立的 timed-intervention 分支。

## Pilot设计

- 按 Qwen transition 分层抽取：5 restored、3 persistent failure、2 worsened、2 both correct。
- 候选介入步：0、4、8、12、16、20、24。
- 目标：A、B、AB。
- 每个样本额外生成/复用 WAIT 原生轨迹。
- 完整 manifest：`manifests/counterfactual_pilot.json`。
- Pilot 所需原生 SS masks 从 `2026_8_31_EXP_2` 的冻结 SAM 结果按 sample ID 打包到
  `segmentation/source_native_masks`，避免依赖任何旧 `/science/...` 目录。

正式批量运行前先缩减为 2 个样本、3 个时刻进行 smoke test，验证 scheduler 分支、同 seed
checkpoint、mask ownership、背景匹配和输出sidecar。

## 必须保存

- 每个决策时刻的原生 checkpoint 哈希及低分辨率预览；
- 每个 `(sample, step, target)` 的最终图与sidecar；
- `outside_write_rms`、`background_native_match_rms`、相对原生状态差；
- 绑定评分、结构评分、运行时间与峰值显存；
- 失败、空mask、OOM和scheduler异常，不得隐藏或补跑覆盖。

## Oracle标签边界

Oracle必须同时考虑文化绑定、结构质量和成本。Qwen绑定标签不能单独决定最优动作；在结构评分
尚未完成前，只能生成候选数据，不能产出正式Oracle标签。

## RL环境冻结接口（v0）

- Observation：当前step、低分辨率预览、实体级内部指标、全局指标、历史工具调用与剩余预算。
- Action：`WAIT / INVOKE(target, rollback_steps) / ACCEPT / ROLLBACK / TERMINATE`。
- Tool：LSDA接收checkpoint、target与回滚跨度，运行局部专家并在固定观察窗口后返回新状态。
- Reward：文化绑定增益、结构变化、泄漏减少、伪影减少和调用成本的分解结果；结构分数低于
  校准门限时触发约束惩罚。
- Pending rule：`INVOKE`后必须先`ACCEPT`或`ROLLBACK`，才能继续`WAIT`或再次调用。

接口实现见 `code/agentic_lsda/rl_env.py`。奖励权重和结构门限尚未校准，不得把测试中的占位参数
用于正式结论；应从离线C1-C2 replay曲线和人类结构标注中确定。
