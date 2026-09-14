# HANDOFF — Result 2：区分「知识性缺失」与「机制性错误」

> 交接对象：新 Agent（负责 Result 2 = 缓解工程）
> 项目：CVPR 2027，SD3.5 MM-DiT 多文化实体生成的属性漂移
> 日期：2026-09-11
> **阅读顺序**：`warning.md`（存储/安全/纪律）→ 本文件 → `code/lsda/README.md`（LSDA 方法）→ `CONVERSATION_SUMMARY.md`（三支柱总览）

---

## 0. 你负责什么（一句话）

> 现有缓解工作**不区分**“模型不认识该文化”（知识性缺失）与“模型认识但组合时串了”（机制性错误），把它们混在一起修，导致**修复结论混杂、恢复率虚高**。你的任务：**把两类失败分开归因、分开处置**，并给出各自的缓解方案与对照实验。

---

## 1. 论文结构与你的位置

标题方向：`Is Attention All You Need?`（待定，见 §8 注意事项）

| 部分 | 内容 | 状态 |
|---|---|---|
| **Result 1（机制）** | 文化绑定失败的因果位点在注意力写回的**内容（Value）**，而非**路由（Attention Weight）** | ✅ 已完成（见 §2） |
| **诊断** | 在线检测 content 污染 / onset | 未开始 |
| **Result 2（缓解，你的任务）** | 区分知识性缺失 vs 机制性错误，分开归因、分开处置 | ⬅ **你负责** |

---

## 2. Result 1 的结论（你必须知道的背景）

**核心发现**（5 对文化实体，N=33 seeds，1024×1024）：

| 指标 | 值 |
|---|---|
| `proj_V`（换内容补上的差距） | **0.82 ± 0.76** |
| `proj_W`（换路由补上的差距） | **0.23 ± 0.85** |
| 比值 | **≈ 3.6×** |

终局图 2×2（pair 1, N=9）：W-fix 5/9、V-fix 6/9（均不显著）；**Both-fix 8/9（p=0.02）**。

**关键概念**：$W=\mathrm{softmax}(QK^\top/\sqrt{d})$ 是**路由**，$V$ 是**内容**；二者**同属注意力**。我们说“不是路由的问题”≠“注意力错了”，而是**在注意力内部把病灶从路由重新定位到内容**。

**证据位置**：`experiment/2026_9_10_EXP_1_CAUSAL/`（Fig 1 = `figures/fig_probe_direction_aware.png`，Fig 2 = `figures/fig_interaction_2x2.png`）。

---

## 3. Result 2 的目标与定义

### 3.1 两类失败的定义

| 类别 | 判据（**纳入门槛**） | 含义 | 处置 |
|---|---|---|---|
| **知识性缺失** (knowledge absence) | standalone A **或** B 就失败 | 模型不认识该文化 | **补充知识**（重训 / 更具体描述 / 换概念） |
| **机制性错误** (mechanism error) | standalone A✓ **且** B✓，但组合 SS✗ | 有知识，组合时内容串了 | **干预修复**（LSDA / 内容级修正） |

**判据的权威定义**见 `code/lsda/README.md` §7：只有“A-alone、B-alone 均正确而原生 SS 失败”的样本，才能用于评价“模型有知识但组合绑定失败”。

### 3.2 核心主张（Result 2 要证明的）

1. `[hypothesis]` 现有缓解工作把两类混在一起，导致**恢复率虚高**（把“本来就该失败的”算进分母/分子）。
2. `[hypothesis]` 分开后：
   - **知识性缺失**用“补充知识”类手段（描述增强/重训）；
   - **机制性错误**用“内容级干预”（LSDA）；
   - 两类各自的恢复率与总体口径不同，且**机制性错误才是 LSDA 的有效靶区**。
3. `[hypothesis]` 若不分流，对知识性缺失样本施加 LSDA 会“无效修复”甚至误导结论。

---

## 4. 现有资产（可直接复用）

| 资产 | 位置 | 说明 |
|---|---|---|
| **100 对文化清单** | `experiment/2026_8_25_EXP_1/cultural100_records/cultural_pairs_100.json` | 每对含 A/B 短描述、长描述、SS/SL/LL Prompt |
| **900 任务历史数据** | `experiment/2026_8_25_EXP_1/cultural100_records/experiment_4500/` | 100 对 × 9 seeds（base×10+replicate）的 SS + LSDA 评分（Qwen+Gemini） |
| **双 VLM 评分工具** | `code/agentic_lsda/score_images.py` | GPT-5.4 + Gemini-3.5-flash（**注意 `reasoning_effort=none`**） |
| **泄漏判定** | `code/mmdit_causal/analyze_deepdive.py` | 严格双裁交集 |
| **LSDA clean v1** | `code/lsda/lsda_pipeline.py`、`generate_lsda_900.py`；定义见 `code/lsda/README.md` | 区域独立 Short-Prompt 专家 |
| **E0 纳入门槛流程** | `code/mmdit_causal/gen_deepdive.py` + `select_seeds.py` | standalone A/B + SS → 分类 |
| **深挖 5 对结果** | `experiment/2026_9_10_EXP_1_CAUSAL/` | 45 seed 中 29 泄漏、7 满足完整纳入门槛 |

---

## 5. 建议的实验设计（供你决定）

### 5.1 第一步：普查分类（回答“两类各占多少”）
- 对 100 对（或子集 20-40 对）× 每对 N 个 seed，生成 **standalone A / standalone B / mixed SS**（1024×1024）。
- 双 VLM 评分 → 分类：
  - `knowledge-absence`：A 或 B 失败；
  - `mechanism-error`：A✓ B✓ 且 SS✗；
  - `both-correct`：A✓ B✓ 且 SS✓。
- 产出：**两类占比表**（这是“混为一谈”问题的量化证据）。

### 5.2 第二步：分开处置
- **机制性错误** → LSDA（内容级局部专家），报告恢复率；
- **知识性缺失** → 描述增强（SL/LL，长描述）或承认需重训，报告其**无法靠干预修复**；
- 对照：对知识性缺失样本也施加 LSDA，展示**无效/误导**。

### 5.3 第三步：口径对比（Result 2 的“虚高”证据）
- 混合口径恢复率 vs 分流后恢复率，展示差异。

---

## 6. 环境与复现

| 项 | 值 |
|---|---|
| 主力服务器 | `ssh -p 36111 wx@s3.v100.vip`（备机 A100，有 SAM） |
| Python | `/science/wx/pry/.venv/bin/python` |
| SD3.5 | `/science/wx/pry/models/stable-diffusion-3.5-large` |
| SAM | `/science/wx/pry/models/sam-vit-base` |
| 代码 | `/science/wx/pry/EXP1/code`（本地镜像 `code/mmdit_causal/`） |
| GPU | 6×A100，用空闲卡；生成前先 `nvidia-smi`，长任务用看门狗 |
| 生成基线 | 1024×1024，28 步，CFG 4.5，fp16 |

---

## 7. 教训与红线（Result 1 踩过的坑，务必避免）

1. **张量维度必须核对形状**：`h_fix` 曾用错 `dim`（在 `[seq,d]` 上用 `dim=1`），512 下静默出错、被误判为“修复”。任何张量操作先 `print(shape)`。
2. **“变了 ≠ 对了”**：度量要**方向感知**（朝正确答案靠近多少），不能只看“变化量”。
3. **VLM 响应可能被截断**：思考型模型必须加 `reasoning_effort=none`，否则内容被截成 `{`。
4. **结论要区分证据等级**：`[data-supported]` / `[hypothesis]` / `[exploratory]`（见 `warning.md`）。
5. **失败样本不隐藏**：知识性缺失样本要单独报告，不能塞进机制性错误的恢复率里——这正是 Result 2 要纠正的。

---

## 8. 开放问题（需你/负责人决策）

1. **普查规模**：100 对全做，还是子集？每对几个 seed？
2. **“补充知识”的合法手段**：SL/LL 描述增强是否算“补充知识”？重训不在当前算力内。
3. **标题**：`Is Attention All You Need?` 有风险（我们其实说注意力内部的内容才是病灶），可考虑 `Routing Is Not All You Need: Content Dominates ...`。
4. **术语**：`routing`/`content` 是我们的操作性定义，Method 里必须显式声明（$W$、$V$ 同属注意力）。

---

## 9. 文件地图（Result 1 交付物，供参考）

| 文件 | 内容 |
|---|---|
| `experiment/2026_9_10_EXP_1_CAUSAL/REPORT.md` | 完整机制报告（含撤回记录） |
| `.../MAIN_RESULT_1.md` | Result 1 正式表述（中英） |
| `.../METHOD_DRAFT.md` | Method 节草稿 |
| `.../EXPERIMENTS_DRAFT.md` | Experiments 节草稿 |
| `.../HANDOFF.md` | Result 1 交接 |
| `.../figures/` | 保留图（Fig 1–6 + demo） |
| `.../_recycle/` | 旧口径 / bug 产物（勿用于正式结果） |

---

## 10. 一句话给你

> 你从 Result 1 的结论（**内容 Value 主导**）出发，证明**现有缓解工作的恢复率因混入知识性缺失而虚高**，并给出“**知识性缺失补充知识、机制性错误内容级干预**”的分流框架与对照实验。先做 §5.1 的普查分类，这是 Result 2 的立足点。
