# 文化绑定漂移：度量、干预与 Agent 决策 —— 研究叙述与三支柱进度总览

> 更新：2026-09-03（修订版：统一失败口径，恢复历史 P1 机制证据，增加 LSDA 转移矩阵、Agent 必要性门槛与统一样本主表）。
> 权威规范：`warning.md`；LSDA 方法：`code/lsda/README.md`；绑定设计附录：`code/agentic_lsda/culture_token_binding.md`。
> **证据纪律（D1）**：所有科学陈述只允许标为 `[data-supported]`、`[hypothesis]` 或 `[exploratory]`。项目进度符号不得替代证据等级。
> **视觉纪律**：视觉结论须来自冻结的人工/VLM 评审记录；工具是否能直接查看图像取决于具体会话能力，不再用“Agent 天生看不见图像”作为一般性说明。

---

## 总述（三支柱叙事）

> **问题**：生成多个文化实体时发生**属性泄漏/错误绑定**（如右瓶"意美奥利卡"被画成左瓶"中青花山水"）。
>
> **① 发现与解释**：先观察到该漂移；进而希望在**去噪过程中截取内部数据、设计指标**衡量是否发生偏移，并解释其成因。
>
> **② LSDA 作为干预工具**：`[data-supported]` LSDA 能恢复大量原生 SS 失败样本，但也会损伤少量原本正确样本。`[hypothesis]` 介入时机存在“过早损伤结构、过晚难以逆转”的权衡；正式窗口仍需由各候选时刻的终局纹理、结构与成本联合验证。
>
> **③ Agent Planner**：把 MM-DiT 整个去噪过程当作 Agent 的**交互环境**，由 Agent 判断：一次生成**是否需要 LSDA 介入、在什么时机、对哪些对象**调用局部专家工具（未来还要选工具）。

三支柱构成待验证的闭环：**① 定义并检测漂移 → ② 用干预验证机制并提供修复工具 → ③ 仅在动态决策确有必要时学习何时、对谁介入**。

## 0.1 主评价口径（冻结）

用户要求每个 seed 独立计分，且只有 **Qwen 与 Gemini 均判失败**才记为失败：

```text
strict_failure = qwen_failure AND gemini_failure
union_failure  = qwen_failure OR  gemini_failure
```

`[data-supported]` 当前 900 个配对 seed 的严格交集转移矩阵为：

| 原生 SS | LSDA | 数量 |
|---|---|---:|
| 失败 | 恢复 | 210 |
| 失败 | 仍失败 | 21 |
| 正确 | 变坏 | 7 |
| 正确 | 仍正确 | 662 |
| 合计 |  | 900 |

据此：

- `[data-supported]` 原生 SS 严格失败率：231/900 = **25.67%**；
- `[data-supported]` LSDA 后严格失败率：28/900 = **3.11%**；
- `[data-supported]` 原生失败中的恢复率：210/231 = **90.91%**；
- `[data-supported]` 原生正确样本中的损伤率：7/669 = **1.05%**。

历史的 `502/900=55.78%` 与 `146/900=16.22%` 可作为更宽的“至少一个评审判失败/非严格正确”口径保留，但**不得再标注为双 VLM 共识失败**，也不得与上述严格交集数字混用。正式论文主端点采用严格交集，宽松口径仅作敏感性分析。

---

# 支柱 ① 发现与解释文化绑定漂移

## 1.1 现象与纳入门槛

- `[data-supported]` 同 prompt 的多文化实体生成会出现属性漂移、错误绑定、融合和结构损伤；严格交集口径下，原生 SS 为 **231/900（25.67%）**。
- `[data-supported]` 同一 prompt 的不同 seed 可产生不同结果，因此每个 seed 必须独立计分，不能按 prompt 多数票把失败 seed 覆盖掉。
- `[data-supported]` 本阶段 8-seed P1 扫描中存在正确与失败轨迹，但“约半数”只适用于该 prompt 与该轮人工观察，不外推到全部文化对。

机制研究的正式纳入门槛为：

```text
standalone A 正确
AND standalone B 正确
AND mixed SS 失败
```

只有这一子集能够支持“模型具有单实体知识，但组合绑定失败”的解释。standalone 失败的样本归入知识/概念不足，不进入内部绑定机制主分析。

## 1.2 度量手段：从"图级检测"到"中程测量"（已做到的部分）
| 度量 | 状态 | 结论 |
|---|---|---|
| 双 VLM 盲评（Qwen+Gemini，外部裁判） | 已完成 900 张主评审及 264 条分支评估 | `[data-supported]` 可提供图像终点标签；主口径为两者均失败 |
| 自动化指标（siglip/clip/dino/csd/blip） | 已完成 | `[data-supported]` 在当前样本和配置上与 VLM 共识 κ=-0.353，不可未经校准直接替代评审 |
| **SFT 诊断器 v6**（Pwen-VL-3B = Qwen2.5-VL-3B + LoRA） | 训练完成 | `[data-supported]` 终局测试 drift AUC **0.773**、leakage AUC **0.831**；可作候选预警分数 |
| **checkpoint 级测量**：264 反事实分支在若干 t 取去噪预览喂 v6 | 已完成 | `[exploratory]` 聚合概率随介入时机变化（native 0.495；t=0 约 0.460；t=24 约 0.487），但逐样本 range 多数较小，尚不能据此定义机制起点或最佳窗口 |

→ `[data-supported]` 终局图级检测已经打通。`[exploratory]` checkpoint 预览包含一定预测信号，但 v6 主要由终局图像训练，中间预览可能属于分布外输入；必须经过 checkpoint 专用校准，才能作为在线 Agent 的可靠观测。

## 1.3 历史 P1 内部机制证据与边界

“内部激活归因从未做过”是不准确的。历史 Phase 2.5–2.8 已对 SD3.5 Large 的 MM-DiT block 插桩，记录：

```text
h_in
  -> attn_raw
  -> attn_gated = gate_msa * attn_raw
  -> h_attn = h_in + attn_gated
  -> f_raw = FFN(modulated h_attn)
  -> f_gated = gate_mlp * f_raw
  -> h_out = h_attn + f_gated
```

`[data-supported, P1 only]` 历史汇总结果主要来自 Chinese blue-and-white × Italian maiolica、约 4 seeds：

| 指标/干预 | 历史汇总 |
|---|---:|
| 文化轴 probe accuracy | 0.932 |
| leave-one-seed-out probe accuracy | 0.9887 |
| 热点 | step 26, layer 36 |
| `E(h_in)` | -0.299 |
| attention gated write | +0.270 |
| `E(h_attn)` | -0.028 |
| FFN gated write | +0.379 |
| `E(h_out)` | +0.351 |
| `attn_raw -> attn_gated` | 0.028 → 0.270（约 10×） |
| `f_raw -> f_gated` | 0.014 → 0.379（约 27×） |
| SL `attn_gated` → SS swap | +0.611，4/4 seeds 同向 |
| `gate_mlp` swap | -0.0001 |
| SS/SL `gate_mlp` RMS 差异 | 0.0019 |
| late-window 前已存在的差异 | +0.297 |

由此只能写：

- `[data-supported, P1 only]` 晚期 attention gated write 携带对终局视觉具有因果效力的内容；
- `[data-supported, P1 only]` attention 与 FFN 分支都能沿文化轴写回较大分量，两个 gate 会放大已有分支写入；
- `[data-supported, P1 only]` 交换 gate 向量本身几乎不能修复终局，故不能把 gate 称为语义来源；
- `[hypothesis]` FFN 轨迹可能是提前预测漂移的有效信号，但历史实验没有证明 FFN 是最早来源，也没有训练该预测器。

历史文化轴建立在匹配 standalone controls 的 **`h_out` 均值差**上，不是 `f_gated` 轴。当前仓库缺少逐 `seed × condition × step × layer × token` 的历史原始激活张量及可重建 source data，因此上述内容是历史汇总证据，必须在当前 Baseline v2 上复现后才能泛化。

完整交接见：`experiment/2026_9_2_EXP_1_MLP_AGENT_HANDOFF/HISTORICAL_MLP_DATA_HANDOFF.md`。

## 1.4 当前待验证的内部流路线：MLP-axis

新实验不再只画无方向的隐藏状态能量，而是计算文化方向明确的 FFN 写入：

\[
f^{\mathrm{gated}}_{L,t}=h^{\mathrm{out}}_{L,t}-h^{\mathrm{attn}}_{L,t}.
\]

对 mixed B 的同器型 controls，在相同 seed、相同模板、相同 ROI 下建立：

\[
\hat a^{FFN}_{L,t}
=
\frac{\mu^{FFN}_{A,L,t}-\mu^{FFN}_{B,L,t}}
{\|\mu^{FFN}_{A,L,t}-\mu^{FFN}_{B,L,t}\|_2}.
\]

再将 mixed SS 目标区域的 `f_gated` 投影到该轴。只有在 held-out controls 冻结阈值后，连续两个 decision points 超出 standalone B 方差区间，才定义为候选 onset。

- `[hypothesis]` MLP gated-write trajectory contains an early predictive signature of later cultural binding drift.
- `[exploratory]` 需要比较该 onset、视觉预览 onset 与 oracle 最佳 LSDA 介入时刻是否相关。
- `[data-supported]` 现存两类 legacy energy 图只能说明分支扩大/缩小隐藏状态能量或 SS–SL 距离，不能说明差异沿 A→B 文化方向，因此不能替代 MLP-axis。

## 1.5 当前竞争解释

- `[data-supported, P1 only]` maiolica standalone 可生成正确，排除了该实例中的“模型完全不认识 B”；不能据此排除其他文化对的知识不足。
- `[hypothesis]` H1：池化文本条件经 AdaLN 调制向所有 image tokens 注入共享语义，可能弱化实体归属。
- `[hypothesis]` H2：早步/粗层先形成错误布局或身份承诺，后续分支在该状态上继续写入。
- `[data-supported, narrow intervention]` 指定中层窗口的 joint-attention mask 没有翻转该样本身份；这只否定该 mask、层窗和实现，不能推出 attention 不是来源或载体。

## 1.6 支柱 ① 小结

`[data-supported]` 终局检测已完成，历史 P1 已有内部激活分解和晚期 causal swap 证据。`[exploratory]` checkpoint 在线检测仍需校准。`[hypothesis]` 最早来源与跨文化对普适机制尚未确定；下一项最有信息量的实验是以 standalone-valid 的正常/异常配对轨迹建立 MLP-axis，并用定向替换验证，而不是重复绘制无方向能量曲线。

---

# 支柱 ② LSDA 作为干预工具（优化 + 工具集）

## 2.1 干预的代价权衡（叙事核心论点，已有实验证据）
- LSDA 早介入 vs 晚介入的权衡已被 **pilot 264 分支时序曲线**支持：
  - 早介入（t≈0）对尚未稳定的布局/轮廓有破坏风险（LSDA 是"结构条件下再生成"，可能产生新结构：历史"变坏 28–68 例"）；
  - 晚介入（t≈24）LSDA 残差变小（曲线右端漂移概率接近 native 0.487）→ 无法逆转已形成的文化绑定。
- 即：**存在一个"介入窗口"，窗口内时机敏感** —— 这正是 Agent 决策层的存在理由。

## 2.2 LSDA 工具现状（量化）
| 项 | 结果 |
|---|---|
| LSDA clean v1（区域独立 Short Prompt 专家，step0 起同噪声） | 900 任务：native 55.78% → **16.22%**（相对降 70.9%，双 VLM 共识）；代价=变坏 28–68 例（口径依赖） |
| LSDA v2（Scene Planner+Early Binding，2026-08-31 pilot） | **放弃**：结构好但属性保持弱于 v1 |
| 审计 | outside_write_rms≈0（区域外只读），固定 mask（形状纠正受限） |

## 2.3 工具优化与"更多工具"（刚起步）
- 优化方向（未做）：软 mask/形状可纠正、减少拼贴感、工具内部的时机窗口参数化。
- **已评估的"免训练替代工具"：注意力绑定（图像 A↔B + 文化文本词元）→ 不合格**（救不回共享属性混淆；且纯 T2I 无 box 时自动分割难——连通域颜色分割塌成 1 块）。
- **候选工具（提出未实现）**：
  - ELA 式 **latent 锚定**（不只 mask，更新隐变量把实体锚进区域）；
  - **P+ 式逐层文本注入**（给右瓶专属文本打到决定"内容身份"的 coarse 层）；
  - **区域级独立条件重生成**（= LSDA 思想的直接化：右瓶区域只用意式描述重roll，绕过全局锚点）。
- 环境约束：当前是纯 T2I（无 depth/canny/box），工具大多要自带实例定位（图级分割可靠获取仍是未解难题）。

## 2.4 支柱 ② 小结
**LSDA 是可用的强工具但非免费**：降漂移 70.9%，代价=结构风险 + 时机敏感（窗口效应已量化）。**工具集目前只有 LSDA 一个合格成员**；绑定被评估淘汰；ELA/P+ 式新工具待实现。负责人提出"优化 LSDA + 引入更多工具"——这是开放方向，尚未系统展开。

---

# 支柱 ③ Agent Planner（何时/对谁/用什么工具）

## 3.1 问题描述
把 **MM-DiT 去噪过程作为 Agent 的交互环境**：每一步可观测（当前步数、预览、诊断信号），Agent 决定**是否需要 LSDA、何时（时机）、对哪些对象（A/B/AB）**调用局部专家工具；调用后可接受或回滚。未来扩展为多工具选择。

## 3.2 进度（到哪了）
| 件 | 状态 |
|---|---|
| Agent 环境接口 v0（`code/agentic_lsda/rl_env.py`） | ✅ 冻结：Observation / Action{WAIT, INVOKE(target,rollback_steps), ACCEPT, ROLLBACK} / Pending 规则 |
| 离线反事实数据（供 Oracle/BC/DPO） | ✅ 264 分支 + 12 WAIT 生成（12 样本×7 时刻×A/B/AB） |
| 诊断器 v6 作观测/预警信号 | ✅ 就绪（AUC 0.77） |
| 分支打分（v6→264） | ✅ 聚合时机梯度真实；**逐样本区分度弱**（range ≤0.13） |
| MVP 计划（是否 Agent 优于固定 t=14） | 📄 已冻结（0.8B→3B VLM 方案，见附录） |
| **策略训练（SFT-2/BC → DPO/RL）** | ❌ **未开始** |

## 3.3 待办 / 缺口
1. **reward 源**：诊断器概率弱（逐样本区分度不够）→ 候选：偏好型 reward（分支两两比较）、诊断器暖启动+偏好接棒；
2. 动作空间从"调 LSDA"扩展到"选工具"（依赖支柱②工具集扩充）；
3. Oracle 标签需要"结构保持"验证器（C2）补齐后才能出正式标签；
4. 网格扩样本（50–80）以确认逐样本时机差异是否存在。

## 3.4 支柱 ③ 小结
**决策层的"零件"基本就绪（环境接口/离线数据/诊断信号），但"训练/验证"一步未迈出**。卡点 = reward 信号弱 + Oracle 标签未定。若支柱②只把 LSDA 当唯一工具，Agent 问题可先收敛为"是否介入 + 何时 + 对谁"的时机问题（当前 MVP）。

---

## 三支柱衔接逻辑

```
生成请求 (SS prompt)
   │
   ▼
[MM-DiT 去噪过程] ── ① 中程检测: 诊断器在 t 时刻判"剩余漂移概率"(已可行)
   │                    │  > 阈值 → 触发 Agent
   ▼                    ▼
[Agent Planner ③] ── 决策 { 介入? 何时 t*? 对谁 A/B? 用哪个工具(未来) }
   │
   ▼
[② 工具: LSDA(可用,有窗口代价) + 候选工具(ELA/P+/区域重生成)] ── 执行
   │
   ▼
[ACCEPT / ROLLBACK] ── 观测结果 → 回到去噪过程 / 结束
```

---

## 附录 A：服务器与配置 —— 其他 Agent 快速上手（必读）

### A0 工作区布局
- 本机工作区：`D:\Python\MMDIT\`（Windows）。本地只放：`code\`（脚本）、`data\`（图像）、`experiment\`（记录）、`models\`（VLM 模型）、`references\`、`warning.md`、`CONVERSATION_SUMMARY.md`。
- SSH 凭据**不在工作区**，位于 `%TEMP%\dsh_mmdit_sshcred\`：`askpass_dell.exe`、`known_hosts_dell`（勿入库；`.gitignore` 已保护）。

### A1 服务器清单

| 角色 | 地址/登录 | 用途 | 要点 |
|---|---|---|---|
| **A100_shuyou_1（主力 SD3.5 生成）** | `ssh -p 44609 dell@frp-use.com` | 所有绑定/扫 seed/概念测试 | GPU1 空闲 ~68GB（GPU0 被他人占用）；见 A2/A3 |
| 旧 A100 | `ssh -p 36111 wx@s3.v100.vip`，密钥免密 | LSDA EXP/pilot 264 分支（历史） | 易断连；SD3.5-large 768/1024+CFG OOM(~79GB)；有 SAM |
| RTX4090（paratera，SFT 用） | `root@ackcs-00gjhlar`（paratera askpass） | Pwen-VL-3B SFT/评估 | transformers **4.52.4** + peft 0.14 + GradScaler；v6 LoRA `/root/sft_out_v6/lora` |

- ⚠️ 三台都出现过掉线：命令一律带 `-o ServerAliveInterval=30`，掉线即重试。
- 旧 A100 与 shuyou 是**不同机器**：v6 诊断器在旧 A100/paratera 侧，SD3.5 生成在 shuyou —— 两边数据用 scp 传。

### A2 SSH/SCP 样板（Windows pwsh，shuyou）
```powershell
$env:DISPLAY = ":0"
$env:SSH_ASKPASS = "$env:TEMP\dsh_mmdit_sshcred\askpass_dell.exe"
$env:SSH_ASKPASS_REQUIRE = "force"
$hk = "$env:TEMP\dsh_mmdit_sshcred\known_hosts_dell"
# 上传脚本
scp -P 44609 -o UserKnownHostsFile="$hk" -o StrictHostKeyChecking=no "D:\Python\MMDIT\code\agentic_lsda\X.py" dell@frp-use.com:/home/dell/Z1x/image_by_pry/X.py
# 跑生成（务必后台执行 + GPU1）
ssh -p 44609 -o UserKnownHostsFile="$hk" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 dell@frp-use.com "cd /home/dell/Z1x/image_by_pry && BIND_SEED=11 BIND_OUT=/home/dell/Z1x/image_by_pry/_out PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 /home/dell/Z1x/conda_envs/sd35_clean/bin/python X.py"
```
- 长任务用 `pwsh` 的 `run_in_background: true`；**进度看服务器输出目录**（`ls`），不要干等 ssh stdout（它被缓冲，延迟很久才吐）。

### A3 环境事实（shuyou）
- Python：`/home/dell/Z1x/conda_envs/sd35_clean/bin/python`（torch 2.5.1+cu118、transformers 4.57.3）；**无 cv2/scipy/skimage**，有 numpy/torch/PIL。
- SD3.5：`/home/dell/models/sd3.5-large`，加载用 `local_files_only=True` + `enable_model_cpu_offload()`。
- 生成参数基线：512×512、28 步、CFG 4.5、seed 由脚本 env 控制；单图 ~26s。
- 脚本 env 变量约定：`BIND_SEED` / `BIND_PROMPT` / `BIND_LAYERS=起:止` / `BIND_OUT` / `BIND_CONTROL=1`(mask全零对照) / `SWEEP_SEEDS` / `SWEEP_PROMPT` / `SWEEP_OUT`。
- FLUX.1 权重**不在任何机器**；`flux` conda env 坏（无 torch）；sd35_clean 只有 Flux2Pipeline（FLUX.2，无权重）→ **DreamRenderer-on-FLUX 现阶段不可跑**。
- SAM：旧 A100 有；shuyou **没有**（实例分割只能用颜色/启发式，白瓶体≈白背景时失败）。

### A4 本机模型资产（诊断器）
- `models/Qwen2.5-VL-3B-Instruct\` = 基座权重；`models/Pwen-VL-3B\lora_v6\` = v6 LoRA（诊断器=基座+该 LoRA，drift AUC 0.773）。
- 本机**没有** torch/transformers/peft → 要跑 v6 需在 paratera/旧 A100 上跑，或先装环境（重）。

### A5 一次典型实验的工作流（对照纪律！）
1. 本地 `code/agentic_lsda/` 写/改脚本 → scp 上传到 shuyou `image_by_pry/`；
2. 后台跑（GPU1）；用 `ls 输出目录` 判断进度（native 先出，bound/ctrl 后出）；
3. scp 下载结果到本地 `data/LSDA_Timed/.../`，命名含实验与 seed；
4. 本地量化：`vision_pixel_diff`（本地 sharp，可靠）看 diff；`vision_present` 把图呈现给负责人目视；
5. 更新 `CONVERSATION_SUMMARY.md` 并标注 ✅(量化/目视) / ⚠️(推断)。

### A6 坑与纪律（血的教训）
- **同 prompt 才可对比**：曾拿不同 prompt 的 native/bound 硬比，误报"救回"，已更正 —— 对照组必须同一 prompt/seed。
- **视觉诚实性**：本会话模型是纯文本，`read_image` 返回"图像已省略"，**看不见像素**；判断"某瓶是中式/意式"必须：`vision_describe`（后端不稳定，常限流/超时）或**负责人目视**。`vision_pixel_diff` 是本地计算、可信。
- SDPA 的 `attn_mask` dtype 必须等于 query（fp16）；mask 用 `-1e4` 抑制（softmax≈0）。
- 不要高频/并行连服务器（MaxStartups 限流会概率性丢连接）；一条命令一次做完。
- CUDA 分配加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

## 附录 B：产物与数据索引

| 产物 | 位置 |
|---|---|
| 绑定 v1 / v3 / v2(废弃) / 扫 seed / 概念测试脚本 | `code/agentic_lsda/{bind_sd35,bind_culture_v3,bind_culture_v2,sweep_native,concept_test}.py` |
| 本轮图像 | `data/LSDA_Timed/2026_9_1_bind/{natives_sweep,seed11,seed11_bw,seed11_v3,concept_test,mojolica}` |
| 历史实验 | `data/LSDA*`、`data/SS*`、`experiment/2026_*_EXP_*`（pilot 264 分支等） |
| v6 LoRA | 服务器 `/root/sft_out_v6/lora`；本地 `models/Pwen-VL-3B/lora_v6` |
| 服务器原始副本 | shuyou `/home/dell/Z1x/image_by_pry/_bind_out,_confused,_mojolica,_concept,_bind_v3_11...` |

## 附录 C：待决策清单（负责人）
1. ✅ 概念测试目视（已确认：Mojolica/Maiolica 单实例 = 意式多色）→ **H3 证伪，指向同现污染（H1/H2）**；
2. 是否先做 A1-1 注意力量测（补支柱①的"内部流归因"缺口；区分 H1 vs H2）；
3. 支柱②优先项：LSDA 优化 vs 引入 ELA/P+ 候选工具 vs 先解决"无 box 自动分割"；
4. 支柱③：先跑通 MVP（固定 t 对比）还是先定 reward 源。

---

*快照 8（三支柱叙事版）：2026-09-03 —— ①发现与解释（检测✅/归因进行中：H3 已证伪→同现污染） ②LSDA 工具（窗口代价已量化，工具集待扩） ③Agent Planner（零件就绪，训练未启动）。*
