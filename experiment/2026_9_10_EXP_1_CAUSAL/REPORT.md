# 2026_9_10_EXP_1_CAUSAL 结果报告

> 目标：验证 C1 —— 文化生图属性漂移的因果位点在**注意力写回的内容（Value）**，而非**路由（Attention weights）**或门控。
> 服务器：`s3.v100.vip:36111`（备机 A100）；代码 `/science/wx/pry/EXP1/code`；GPU3/GPU5。
> 本地备份：`data/Causal/2026_9_10_EXP_1/`、`experiment/2026_9_10_EXP_1_CAUSAL/`。

## 0. 设置

- 模型：SD3.5-large，512×512，28 步，CFG 4.5，fp16，math SDPA。
- 文化对：pair1 = Chinese blue-and-white porcelain vase × Italian maiolica vase。
- Donor 语义：**同一次前向内** B-条件分支（batch 3），B 物体置于与混合场景相同位置（右侧），位置对齐、无重采样。
- 2×2 因子：baseline / w_fix（换路由）/ v_fix（换内容）/ both_fix（全换）。

## 1. E0 样本构建与纳入门槛

- 生成 40 seeds × 3 条件（standalone A / standalone B / mixed SS）= 120 图。
- 双 VLM（Qwen3-VL-235B + Gemini-3.5-flash-lite）盲评。
- 纳入门槛：A✓ 且 B✓ 且 SS✗（双裁判均 < 0.5）。
- **结果：13 个已评分 seed 中 9 个合格**（1001, 1005-1012）。

## 2. E1 复现

- MM-DiT block 恒等分解在 L36/t26 验证通过：
  - `attn_gated = gate_msa * attn_raw` 相对误差 **2.07e-4**
  - `f_gated = gate_mlp * f_raw` 相对误差 **2.07e-4**
  - `h_out = h_attn + f_gated` 相对误差 **2.25e-4**
  （fp16 精度内）
- 逐层跨实体注意力（DreamRenderer 式）：top 层集中在 **L0-10**（+ L35）。

## 3. E2 因果（核心）

### 3.1 终局图 2×2（单点 L36/t26，N=9）

| 臂 | 恢复量 Δ | 翻转 seed 数 | sign p |
|---|---|---|---|
| baseline | 0.000 | 0/9 | — |
| w_fix | +0.094 | 5/9 | 0.50 |
| v_fix | +0.072 | 5/9 | 0.50 |
| **both_fix** | **+0.183** | **8/9** | **0.0195** |

- 单点干预：**both_fix 显著**，单独 W 或 V 均弱。route 40% / value 60%（相对贡献）。
- 注：Qwen 裁判存在地板效应（几乎所有臂判 0.0），削弱了信号。

### 3.2 终局图 2×2（窗口 L33-36 × t24-27，N=9）

- 恢复量更弱（+0.011 / +0.022 / +0.036）——多层级 patch 与 donor 状态不一致，效果反而不如单点。

### 3.3 内部探针（**主结果**，N=9，4 层 × 4 步 = 16 点/seed）

在同一次前向内，对目标区域直接计算：
- `O_mixed = W_mixed · V_mixed`
- `O_wfix = W_donor · V_mixed`
- `O_vfix = W_mixed · V_donor`

测"改路由"与"改内容"各自对输出的改变量：

| 指标 | 均值 ± 标准差 |
|---|---|
| route share | **34.1% ± 3.7%** |
| value share | **65.9% ± 3.7%** |

**9/9 seeds 一致**：value 贡献约为 route 的 **1.93 倍**。

## 4. 结论

1. **Value 与 Route 都参与**属性漂移，不是单一因素。
2. **Value 主导**（~66% vs ~34%），且高度一致（std 3.7%）。
3. 单独修任一项在终局图上都不足以稳定翻转；**同时修才显著**（both_fix 8/9, p=0.02）。
4. 这支持 C1 的方向（内容更重要），但**不是"route 无用"**——route 约占 1/3。

## 5. 产出图（均通过 gpt-5.6-sol 美学审计：无重叠，9/10）

| 图 | 内容 |
|---|---|
| **`fig_main_result.png`** | **汇总主图**：(A) route/value 责任占比 + (B) 终局 4 臂恢复量 |
| `fig_internal_share.png` | 逐 seed route/value 责任占比（9 seeds） |
| `fig2_pair1_recovery.png` | 终局图 4 臂恢复量（窗口版） |
| `fig3_pair1_scatter.png` | 逐 seed W-fix vs V-fix 散点 |
| `fig4_pair1_responsibility.png` | route vs value 堆叠责任条 |

## 6. 局限与待办

- 仅 pair1（单文化对）；需 E3 泛化到其余文化对。
- Donor 为同前向 B-条件分支的区域替换，非独立 standalone 的 token 级替换。
- Qwen 地板效应：终局评分信号弱，建议后续改用**比较式 VLM 提问**或内部指标为主。
- 窗口版反而更弱，原因待查（多层级 donor 状态不一致假设）。
