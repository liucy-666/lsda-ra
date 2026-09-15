# code

源代码与启动脚本，按功能分子目录；每个子目录保留一份 README 说明用途。

| 目录 | 用途 |
|---|---|
| `lsda/` | LSDA clean v1 及 v1.1–v1.4 变体、批量驱动、helpers（方法定义见 `lsda/README.md`） |
| `agentic_lsda/` | Agent 反事实数据管线 + RL 环境接口（支柱 ③） |
| `mmdit_causal/` | 受控 100 对的生成 / 双 VLM 评分 / 机制分析（Result 1、EXP_3/6） |
| `metrics/` | 五指标（VLM/纹理/Gram/k-NN/SigLIP）实现与汇总 |
| `culture_comp/` | 文化组合 benchmark 与大尺度实验编排（EXP_5/6，EXP_8 复用） |
| `kb/` | 外置文化知识库构建管线 |
| `vqa/` | 900 任务 Qwen VQA 评分与报告 |
| `cultural100/` | 受控 100 对构造脚本 |

纪律：代码不得硬编码已删除的旧路径；正式方法与探索版本分开命名，不静默覆盖；修改方法数据流/Prompt/mask/评价口径须新建版本并在实验目录记录。
