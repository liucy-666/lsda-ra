# 文化绑定漂移：度量、干预与 Agent 决策 —— 研究叙述与三支柱进度总览

> 更新：2026-09-03（按负责人叙事定稿："发现与解释 → LSDA 干预工具 → Agent Planner" 三点叙述）。
> 权威规范：`warning.md`；LSDA 方法：`code/lsda/README.md`；绑定设计附录：`code/agentic_lsda/culture_token_binding.md`。
> 诚实性：凡"视觉"结论必须负责人目视为准（本 Agent 纯文本，曾误读图像）；本文件标 ✅(机器量化) / ✅(负责人确认) / ⚠️(推断待证)。

---

## 总述（三支柱叙事）

> **问题**：生成多个文化实体时发生**属性泄漏/错误绑定**（如右瓶"意美奥利卡"被画成左瓶"中青花山水"）。
>
> **① 发现与解释**：先观察到该漂移；进而希望在**去噪过程中截取内部数据、设计指标**衡量是否发生偏移，并解释其成因。
>
> **② LSDA 作为干预工具**：LSDA 能部分恢复目标对象的纹理/风格，但**过早介入会破坏尚未稳定的布局轮廓，过晚介入又无法逆转已形成的文化绑定**（已有实验验证）；因此需优化 LSDA，并引入/评估更多工具。
>
> **③ Agent Planner**：把 MM-DiT 整个去噪过程当作 Agent 的**交互环境**，由 Agent 判断：一次生成**是否需要 LSDA 介入、在什么时机、对哪些对象**调用局部专家工具（未来还要选工具）。

三支柱构成闭环：**① 检测/解释漂移 → ② 提供干预工具（LSDA+更多）→ ③ 决策何时/对谁/用什么工具介入**。

---

# 支柱 ① 发现与解释文化绑定漂移

## 1.1 现象（已确认）
- 同 prompt 多文化实体生成存在漂移/漏配/泄漏/融合。历史大样本：native 漂移率（双 VLM 共识口径）**55.78%**（502/900）。
- 本阶段同 prompt 扫 8 seed：**约半数**把右瓶画成中青花山水 ✅(部分由负责人目视复核)。

## 1.2 度量手段：从"图级检测"到"中程测量"（已做到的部分）
| 度量 | 状态 | 结论 |
|---|---|---|
| 双 VLM 盲评（Qwen+Gemini，外部裁判） | ✅ 900/264 级完成 | 共识口径能判漂移；宽松评分判别力不足（H2 不可用） |
| 自动化指标（siglip/clip/dino/csd/blip） | ✅ 完成 | **判废**：与 VLM 共识 κ=-0.353 → 不能当代理/信号，除非先校准 |
| **SFT 诊断器 v6**（Pwen-VL-3B = Qwen2.5-VL-3B + LoRA） | ✅ 训练完成 | drift AUC **0.773**、leakage **0.831**；可当"预警打分器" |
| **中程（checkpoint 级）测量**：264 反事实分支在 t∈{0..24} 取去噪预览喂 v6 | ✅ 完成 | 得到"剩余漂移概率随介入时机变化"的曲线：越早介入漂移越低（native 0.495 → t=0 ~0.46 → t=24 ~0.487）→ **漂移可被中程探测** |

→ **结论：漂移的"检测"已打通两条路——图级（诊断器/盲评）与中程（checkpoint 预览 + 诊断器）。**

## 1.3 内部数据流归因：能确定偏移原因吗？（目前的缺口）
**尚不能。** 进度与缺口：
- ⚠️ **注意力量测（A1-1 critical-layer 分析 / 激活级 hook）未跑成**（早前 OOM 搁置）→ 从 attention/特征流**直接归因**还没做过。
- **间接证据**（绑定实验反推）：在 joint-attention 中层屏蔽跨实例（图像 A↔B，甚至 + 文化文本词元）**翻不动右瓶文化身份**（负责人目视：仍中式；干预量 diff 65.77% 也无效）⇒ 关键语义不走"可被 mask 的中层注意力"通道。
- **归因假设（H3 已证伪，H1/H2 存活）**：
  - ✅ **H3 概念弱/欠定 —— 已证伪**：单实例概念测试（负责人目视）——"Italian Mojolica/Maiolica vase" **单独**生成即是**意式多色**（正确），说明 SD3.5 完全认识该概念。故失败不是"不认识意式"，而是**同现污染**。
  - **H1 池化全局条件通道**：SD3.5 每 block 有池化文本嵌入调制（adaLN/scale-shift），把整句（含 "Chinese blue-and-white porcelain"）摘要灌进两瓶每 token，绕过注意力；
  - **H2 早层/早步定身份**：扩散早期步 + coarse 层已定"右区域=青花"，后期 mask 只动纹理不动身份；
- **已确认的因果链条**：概念单独正确 → 与中青花同现即被污染 → 中层 joint-attention mask（图像+文本词元）翻不动身份 ⇒ 污染通道大概率在 **H1 全局/池化条件** 或 **H2 早步粗层**，而非可被中层注意力 mask 拦截的通道。这为"通道级归因"论文点提供了直接证据链（概念单独正确对照 + 绑定无效对照）。

## 1.4 支柱 ① 小结
**检测：做完了（图级 + checkpoint 中程）。解释：没做完**——只有假设 + 间接证据，缺"内部激活级"归因实验（A1-1 注意力量测是第一步）。这也与叙事①的"设计指标衡量偏移"吻合：图级指标已可用，**内部流指标待建**。

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
