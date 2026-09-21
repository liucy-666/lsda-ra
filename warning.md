# MMDIT 工作区存储规范与警告

> 本文件是 `D:\Python\MMDIT` 的目录治理规则。所有后续实验人员、自动化 Agent 和后台脚本在创建、移动、修改或删除文件前都必须先阅读本文件。

## 1. 根目录结构

根目录的业务内容只使用以下三个文件夹：

```text
D:\Python\MMDIT\
├── code\
├── data\
└── experiment\
└── temp\
```

`.claude`、`.codex` 等点号目录属于工具配置，不是实验数据目录，不得随意移动或删除。

除本规范文件外，不应在根目录堆放脚本、图像、JSON、日志、压缩包、模型权重或临时文件。

---

## 2. `code`：代码与方法文档

### 允许放置

- Python、PowerShell、Shell、SFTP 等源代码和启动脚本；
- 与代码直接对应的 `README.md`、方法说明和必要注释文档；
- 按方法或功能建立的子目录，例如：

```text
code\lsda\
code\vqa\
code\workspace\
code\cultural100\
```

### 禁止放置

- 生成图像、匿名评审图或可视化结果；
- 实验 Prompt、任务清单、design matrix、sidecar JSON；
- CSV、JSONL、VLM 原始回复、统计结果；
- 运行日志、PID、watchdog 状态；
- 模型权重、下载缓存、压缩包和安装程序；
- API key、token、密码或任何凭据。

### 特别要求

- 代码不得继续硬编码已删除的旧路径，例如 `D:\Python\MMDIT\AAA_Experiment` 或根目录旧 `LSDA`；
- 正式方法与探索版本必须分开命名，不能静默覆盖；
- 修改方法的数据流、Prompt 来源、mask 来源、扩散步范围或评价口径时，必须创建新版本并在对应实验目录记录变更；
- `code\lsda\README.md` 是当前 LSDA clean v1 的方法定义，修改 LSDA 前必须先阅读。

---

## 3. `data`：只存放图像

### 允许放置

`data` 中只能存放图像文件，并按“方法/条件 → 实验编号”分层：

```text
data\Standalone_A\2026_8_25_EXP_1\
data\Standalone_B\2026_8_25_EXP_1\
data\SS\2026_8_25_EXP_1\
data\SL\2026_8_25_EXP_1\
data\LL\2026_8_25_EXP_1\
data\LSDA\2026_8_25_EXP_1\
data\Blind\2026_8_25_EXP_1\
```

允许的常用扩展名为 `.png`、`.jpg`、`.jpeg`、`.webp`。

### 禁止放置

- JSON、JSONL、CSV、Parquet、Markdown、TXT；
- Prompt、seed 清单、blind map、评分或统计表；
- Python/PowerShell/Shell 代码；
- 日志、PID、错误记录；
- TAR、ZIP 等压缩包；
- 模型文件或环境依赖。

### 特别要求

- 每张图像的 Prompt、seed、模型配置和方法版本必须写入 `experiment` 下的 manifest/sidecar，不得靠文件名猜测；
- 同一方法的正式比较必须使用冻结且一致的 seeds；
- 不得覆盖已有图像。需要重跑时创建新实验编号或新版本文件；
- 匿名盲评图与原始图分目录保存，blind map 不得放入 `data\Blind`。

---

## 4. `experiment`：实验协议、元数据、结果和日志

### 目录命名

每个新实验必须创建独立目录：

```text
YYYY_M_D_EXP_N
```

示例：

```text
experiment\2026_8_25_EXP_1\
experiment\2026_8_25_EXP_2\
```

同一天的新实验递增 `N`，不得把不同目的的实验混入同一目录。

### 允许放置

- 协议、冻结清单、版本说明和 README；
- Prompt、seed、任务矩阵、配置和 manifest；
- 每张图对应的 sidecar；
- SAM masks 的索引和分割审计记录；
- VLM blind map、rater orders、原始评分、错误记录；
- CSV、JSON、JSONL、Parquet 等派生数据；
- 统计结果、source data、报告和科研图表；
- 运行日志、PID、watchdog 状态；
- 偏差记录、失败案例清单和复现实验队列。

### 禁止放置

- 大批正式生成图像；图像应存放在 `data`；
- 方法源代码；代码应存放在 `code`；
- API key、HF token、SSH 密码或其他秘密；
- 来源不明、无法追溯 Prompt/seed 的结果；
- 不同实验版本共用同一个可变结果文件并互相覆盖。

### 推荐子结构

```text
experiment\YYYY_M_D_EXP_N\
├── protocol\
├── manifests\
├── sidecars\
├── segmentation\
├── ratings\
├── logs\
├── source_data\
├── figures\
└── report\
```

---
## 5. `temp`：临时脚本

### 目录命名

每个Agent开的脚本必须创建独立目录：

```text
agent_name
```

示例：

```text
experiment\GPT-5.6_sol\
experiment\Deepseek_v4_flash\
```

同一Agent脚本在子文件夹下递增 `N`，不得把不同目的的实验混入同一目录。

### 禁止放置

- 大批正式生成图像；图像应存放在 `data`；
- 方法源代码；代码应存放在 `code`；
- API key、HF token、SSH 密码或其他秘密；
- 来源不明、无法追溯 Prompt/seed 的结果；
- 不同实验版本共用同一个可变结果文件并互相覆盖。

---
## 5. 原始数据与结果纪律

1. **V1 原始数据只增不改。** 原图、原始 activation、原始 VLM 回复和初始日志不得在分析时改写。
2. **分析使用副本。** 清洗、去重、聚合和统计应产生新的 derived/source-data 文件。
3. **逐图可追溯。** 每个图表点必须能追溯到 image ID、pair、seed、condition、method 和 rater。
4. **逐 seed 计分。** 不得把三个 seeds 合并成一个 Prompt 级“成功”；必须报告成功图像数、失败图像数及对应 seeds。
5. **失败不得隐藏。** 属性漂移、几何损伤、SAM 错分、实体合并、VLM 分歧和 API 失败都必须保留。
6. **假说不等于结论。** 无冻结数据或因果干预支持时，只能标注为 hypothesis/exploratory。
7. **不得用补跑掩盖失败。** 只有预先规定的复现、无效输出补齐或明确的新版本实验可以重跑。

---

## 6. 后台任务与看门狗

- 启动生图、下载或 VLM 评审前，必须记录脚本、参数、并发数、PID 和日志路径；
- 远程长任务必须使用 `setsid nohup command </dev/null >log 2>&1 &`，不得让
  stdout/stderr 或 stdin 继续依附 SSH 通道；启动后必须立即验证 PID 存活、日志文件存在；
- 看门狗只能恢复缺失任务，不得重复已成功的 image ID；
- 断点续跑必须扫描全局有效结果，而不是只检查当前 chunk；
- API 连续失败时应降低并发并保留错误，不得无限创建新 worker；
- 用户要求“停止”或“停止生图”时，必须立即停止对应进程，不能只停止监控界面；
- watchdog、worker PID 和运行状态必须保存在对应 `experiment` 目录。

---

## 7. 凭据与远程服务器

- `TEST_API_KEY` 只能在运行时从 Windows 用户环境变量读取，禁止写入代码、日志、JSON、Markdown 或仓库；
- HF token 只能保存在获准位置，不得复制进本地实验目录；
- SSH 密码、旧服务器凭据和带签名的临时 URL 不得写入长期文件；
- 远程实验只允许写入用户指定工作区；其他远程目录按只读处理；
- 本地和远程副本必须通过 manifest、文件数或哈希建立对应关系后，才能删除源副本。
- `s3.v100.vip:36111` 存在公网 pre-auth 连接拥塞；客户端默认使用
  `ConnectionAttempts=8`、`ConnectTimeout=10`、`ServerAliveInterval=10`、
  `ServerAliveCountMax=6`，并减少并行短连接、尽量合并传输。
- Windows OpenSSH 在本机实测不支持 ControlMaster（`getsockname failed: Not a socket`），
  因此不得把 Unix 的 `/tmp/ssh-ctrl-*` 配置直接复制到 Windows 客户端。
- 修改共享服务器的 `MaxStartups`、`LoginGraceTime`、fail2ban 或防火墙属于全局管理操作，
  必须由服务器管理员授权并先通过 `sshd -t`；普通实验 Agent 不得自行修改或重启 SSHD。

---

## 8. 删除、移动与清理

- 删除前必须明确目标的绝对路径，禁止对工作区根目录使用递归删除；
- 必须先确认文件已迁移且新副本可读，再删除旧目录；
- 未经用户明确授权，不得删除模型、原始数据、代码仓库或大批实验文件；
- 不得通过换命令、换 shell 或间接脚本绕过删除审批；
- 临时文件应写入系统临时目录或实验专属临时子目录，任务完成后按授权清理；
- 新工作者发现不确定文件时，应先记录和询问，不得凭名称猜测后删除。

---

## 9. 新实验启动检查清单

开始新实验前确认：

- [ ] 已创建新的 `experiment\YYYY_M_D_EXP_N`；
- [ ] 代码写入 `code`，图像目标写入 `data`；
- [ ] Prompt、seed、配置和主要对比已冻结；
- [ ] standalone 与组合条件使用可比 seeds；
- [ ] 输出不会覆盖旧实验；
- [ ] sidecar 和 manifest 路径已设置；
- [ ] 后台进程、日志和 watchdog 路径已设置；
- [ ] VLM 评分规则和盲化映射已冻结；
- [ ] 未在任何文件中写入密钥；
- [ ] 已阅读相关方法目录中的 README。

违反本规范生成的数据必须在报告中标记为 protocol deviation；严重情况下不得进入正式统计。

---

## 10. 重复问题停止规则（实验负责人 2026-08-31 要求）

**当一个类似的问题重复出现三次未能解决时，Agent 必须及时停止当前操作并立即向实验负责人汇报，不得继续重试或绕路。**

适用情形包括但不限于：SSH/网络连接反复被掐断、API 连续失败、同一脚本反复报错、文件访问反复被拒、服务器资源反复不足等。

执行要点：

- "重复出现三次"指同类问题（同一根因、同一现象）连续或累计出现 3 次仍未解决；
- 停止后必须汇报：问题现象、已尝试的手段（含次数）、当前状态、以及你认为需要的决策或授权；
- 汇报后等待负责人指示，不得自行决定继续重试、换命令、换方式绕过；
- 本规则优先于"完成任务"的目标：连接不稳时宁可暂停，也不得通过高频重试加重服务器负担。

违反本规则的 Agent 行为将被视为 protocol deviation 记录。

---

## 11. 关键配置路径与运行注意（2026-09-15 更新）

- **本地工作区**：`D:\Python\MMDIT`；Windows / PowerShell 5.1；Python 3.12；RAM 15.2GB → **大模型必须串行单进程**。
- **服务器（主力 SD3.5 生成）**：`ssh A100`（= `s3.v100.vip:36111`，user `wx`）
  - python：`/science/wx/pry/.venv/bin/python`
  - 模型：`/science/wx/pry/models/{stable-diffusion-3.5-large,sam-vit-base}`
  - 代码：`/science/wx/pry/MMDIT/code/...`；实验：`/science/wx/pry/MMDIT/experiments/...`
  - 生图参数：SD3.5-large，1024×1024，28 steps，CFG 4.5，fp16。
- **历史/备用服务器**：shuyou `ssh -p 44609 dell@frp-use.com`（已较少使用；详细交接已归档 `tmp/`）。
- **本地模型**：`models/Qwen2.5-VL-3B-Instruct`（基座）、`models/Pwen-VL-3B/lora_v6`（诊断器 v6）。
- **凭据**：`TEST_API_KEY` 仅从 Windows 用户环境变量读取，**禁止落盘**；VLM 评分必须 `reasoning_effort=none`。
- **回收站**：`tmp/recycle_2026_09_15/`，其中 `MOVES.tsv` 逐条记录「原路径 → 去向」，可还原；`tmp/` 已 gitignore，**不得提交 GitHub**。
- **Git**：remote `https://github.com/liucy-666/lsda-ra.git`；本地代理 `http://127.0.0.1:7892`。
- **忽略项**：`.venv/`、`models/`、`data/`、`tmp/`、`**/__pycache__/` 不入库。
- **网络**：Wikidata API 可达；Wikipedia / Google / GitHub / raw.githubusercontent / huggingface 被墙（codeload、jsdelivr 可达）。
- **外部辅助 Agent：Pi**（命令 `pi`，全局 npm 包 `C:\Users\admin\AppData\Roaming\npm\pi.ps1`）
  - 终端可调用的 AI coding assistant，自带 `read/bash/edit/write` 工具；**上下文占用小、支持自动扩展**。
  - 非交互用法：`pi -p "<指令>"`；可指定 `--provider <name>`、`--model <pattern>`、`--thinking <level>`、`--mode json`、`--no-session`。
  - 用途：把**机械/低价值杂活**外包给它，主线 Agent 保留上下文——如批量重命名、格式转换、目录清单、日志抓取、简单脚本、数据统计等。示例：
    ```
    pi -p "把 D:\Python\MMDIT\data\X 下所有 png 转成 1024 jpg 到 Y，并输出文件数"
    ```
  - 约束：Pi 同样会读写文件，**必须遵守本文件全部纪律**；不得让它接触 API key/凭据、原始数据、正式结果，或执行删除/提交等敏感操作；产出需主线 Agent 复核后才能进入正式资产。
