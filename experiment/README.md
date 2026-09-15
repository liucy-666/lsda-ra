# MMDIT experiment layout

- `D:\Python\MMDIT\code`：源代码与启动脚本。
- `D:\Python\MMDIT\data`：仅图像，按生成方法 / 实验 ID 分层（已 gitignore）。
- `D:\Python\MMDIT\experiment`：协议、manifest、sidecar、评分、分析、图表与源数据。
- `D:\Python\MMDIT\tmp`：回收站（`recycle_2026_09_15/`，含 `MOVES.tsv` 可还原；已 gitignore）。

## 命名与文档规则

- 每个新实验使用独立目录 `YYYY_M_D_EXP_N`（同一天递增 N），不得混入不同目的的实验。
- **每个实验目录下有且仅有一份 `README.md`**，记录：实验目的、数据/方法、结论、进度、未来方向。
- 图像不得放入 `experiment/`；代码不得放入 `experiment/`；原始数据只增不改。

## 当前实验索引

| 目录 | 内容 |
|---|---|
| `2026_8_25_EXP_1` | LSDA clean v1 全量 900 任务基线 |
| `2026_9_1_EXP_1` | Agent 反事实时机 Pilot + SFT 诊断器 |
| `2026_9_10_EXP_1_CAUSAL` | Result 1：内容 V 主导路由 W |
| `2026_9_12_EXP_2_LSDA_SOFT` | LSDA 版本消融（v1.1→v1.4） |
| `2026_9_12_EXP_3_KA_ME` | 受控 100 对：统一知识注入 + 局部重扩散 |
| `2026_9_14_EXP_4_CULTURE_KB` | 外置文化知识库 |
| `2026_9_14_EXP_5_CULTURE_COMP` | 文化组合 benchmark |
| `2026_9_15_EXP_6_KNOW_INJECT` | 知识注入位置（LL vs LSDA） |

总体方向与进度见仓库根 `README.md`；目录/数据/凭据纪律见 `warning.md`。
