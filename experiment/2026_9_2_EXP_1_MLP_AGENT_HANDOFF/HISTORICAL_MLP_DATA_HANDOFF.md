# 历史 MLP/FFN 机制数据交接

## 1. 数据状态

- `[data-supported]` 历史 Phase 2.5–2.8 对 SD3.5 Large 的 MM-DiT block 做过插桩，记录过 `h_in`、`attn_raw`、`attn_gated`、`h_attn`、`f_raw`、`f_gated`、`h_out`。
- `[data-supported]` 历史文化对比轴建立在匹配 standalone controls 的 **`h_out` 均值差**上，不是建立在 `f_gated` 上。
- `[data-supported]` 当前整理后的本地仓库未发现逐 `seed × condition × step × layer × token` 的历史原始激活张量，也未发现 `ownership_trajectory_summary.csv` 等原始派生表。
- 因此，本文件中的数值只能作为历史汇总证据和新实验设计依据，不能替代新路线所需的 MLP-axis 原始数据。

## 2. 历史模块定义

```text
h_in
  -> attn_raw
  -> attn_gated = gate_msa * attn_raw
  -> h_attn = h_in + attn_gated
  -> f_raw = FFN(modulated h_attn)
  -> f_gated = gate_mlp * f_raw
  -> h_out = h_attn + f_gated
```

真正写回残差流的 FFN 分量为：

\[
f_{\mathrm{gated}}=h_{\mathrm{out}}-h_{\mathrm{attn}}.
\]

## 3. 历史汇总结果

研究对象为 P1 历史主案例：Chinese blue-and-white 与 Italian maiolica；历史样本量主要为 4 seeds，故不能直接外推到其他文化对。

| 项目 | 历史汇总值 |
|---|---:|
| 文化轴 probe accuracy | 0.932 |
| leave-one-seed-out probe accuracy | 0.9887 |
| 热点 | step 26, layer 36 |
| SS `E(h_in)` | -0.299 |
| SS attention gated write | +0.270 |
| SS `E(h_attn)` | -0.028 |
| SS FFN gated write | +0.379 |
| SS `E(h_out)` | +0.351 |
| `attn_raw -> attn_gated` | 0.028 -> 0.270，约 10 倍 |
| `f_raw -> f_gated` | 0.014 -> 0.379，约 27 倍 |
| SS-SL attention-write difference | +0.196，t=4.34 |
| SS-SL gate-amplification difference | +0.215，t=5.46 |
| SS-SL FFN-write difference | +0.217，t=5.85 |
| gate_mlp swap correction | -0.0001 |
| SS/SL gate_mlp RMS difference | 0.0019 |
| SL attn_gated -> SS swap correction | +0.611，4/4 seeds 同向 |
| late-window pre-existing difference | +0.297 |
| endpoint `mlp_gate_b075` improvement | +0.0002，约 1.1 SEM |

历史解释边界：

- `[data-supported, P1 only]` `f_gated` 在热点处具有较大的文化轴投影写入。
- `[data-supported, P1 only]` gate 剂量能改变内部轴投影，但单独压缩 MLP gate 的端到端视觉修复接近零。
- `[data-supported, P1 only]` 注意力 gated write 互换具有明显端到端修复作用，gate 向量互换没有。
- `[hypothesis]` FFN trajectory 可能是漂移预测器的有效观测，但历史实验没有训练或验证过这种预测器。

## 4. 现存图像

1. `../2026_9_1_EXP_1/figures/legacy_absolute_hidden_energy_by_step.png`
2. `../2026_9_1_EXP_1/figures/legacy_condition_difference_energy_by_step.png`
3. `../2026_9_1_EXP_1/figures/legacy_step00_27_attention_ffn_total.png`

这些图可以用于回顾现象，但目前缺少可重建它们的 source-data 表。

## 5. 两类能量图的含义

根据图轴名称和历史 block 恒等分解，可重建其设计逻辑如下；由于绘图脚本和源表缺失，以下公式标记为 `[historical reconstruction]`，新实验必须重新实现并验证。

### 绝对隐藏状态能量变化

对每个 step，在 block 内计算：

\[
\Delta A_L=\|h_{\mathrm{attn},L}\|_2^2-\|h_{\mathrm{in},L}\|_2^2,
\]

\[
\Delta F_L=\|h_{\mathrm{out},L}\|_2^2-\|h_{\mathrm{attn},L}\|_2^2.
\]

沿 block 累积：

\[
C_A(L)=\sum_{k\le L}\Delta A_k,
\quad
C_F(L)=\sum_{k\le L}\Delta F_k,
\quad
C_T(L)=C_A(L)+C_F(L).
\]

该图回答模块增加或减少多少隐藏状态能量，不回答变化是否朝向 A 或 B。

### 条件差异能量变化

先对同 seed 的两条条件轨迹做差，例如：

\[
d_{\mathrm{in}}=h_{\mathrm{in}}^{SS}-h_{\mathrm{in}}^{SL},
\]

然后把上式中的 `h` 换成 `d`，计算 attention/FFN 对 **SS-SL 距离能量**的增减并沿 block 累积。

该图回答哪个分支扩大或收缩条件间差异，但仍不回答差异是否是文化泄漏、结构差异或其他语义差异。

因此，这些能量图不能替代文化轴投影。

## 6. 对 Agent 所提“MLP-axis 路线”的判定

该路线是合理的新实验，但需要修改两点：

1. `standalone_A` 和 `standalone_B` 必须与 mixed B 使用同一器型、同一空间模板和同一 ROI。例如 mixed B 是 vase，则 controls 必须是 A-culture vase 与 B-culture vase；不能直接拿 A bowl 与 B vase 建轴。
2. “首次越过方差区间”容易受单点波动影响。最小稳健规则建议为：在 held-out controls 冻结阈值后，连续两个 decision points 越界才定义为 onset。

MLP-axis 定义：

\[
\hat a^{FFN}_{L,t}
=
\frac{\mu^{FFN}_{A,L,t}-\mu^{FFN}_{B,L,t}}
{\|\mu^{FFN}_{A,L,t}-\mu^{FFN}_{B,L,t}\|_2},
\]

其中均值对象是目标 ROI 的 `f_gated`，不是 `h_out`。

该实验完成前只能写：

> `[hypothesis]` FFN gated-write trajectory contains an early predictive signature of later cultural binding drift.

不能写：

> FFN-axis 已经被历史实验验证为机制起点或最优 LSDA 触发信号。

