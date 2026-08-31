# MMDIT Cultural Binding — 工作区总结与方法背景

> 本文件由 2026-08-31 的 Agent 会话总结，供**新对话/后续工作者**快速接续。
> 权威规范以 `warning.md` 为准；LSDA 方法定义以 `code/lsda/README.md` 为准。
> 任何新实验前必须先读这两个文件。

---

## 1. 项目背景

本仓库研究 **SD3.5 Large（MM-DiT）在多文化实体组合生成中的属性绑定问题**：
模型能单独生成文化实体（如中国青花瓷 vs 越南 Bát Tràng 陶瓷），但在 Short–Short（SS）
组合场景中，一个实体的文化纹理、色彩体系、材料观感、装饰纹样会泄漏/污染另一个实体。

研究计划两条主线：
- **机制分析**：追踪 MM-DiT attention/FFN/gate/残差流的跨层与跨扩散步变化（代码尚未收录，不可推断可复现）；
- **方法验证**：**LSDA（Local Specialist Diffusion / 局部专家扩散）**——在 SS 场景轮廓内为各实体分配独立 Short Prompt 专家，降低属性混合（当前正式方法）。

---

## 2. LSDA 方法（正式版：clean v1）

定义与边界见 `code/lsda/README.md`。核心数据流（`code/lsda/lsda_pipeline.py`）：

1. 组合 SS Prompt + seed 生成原生 SS 去噪轨迹与最终图；
2. SAM 在原生 SS 图上分割各实体，重叠轮廓仲裁为**严格互斥 one-hot owner masks** + 背景补集；
3. 所有专家从同一初始 latent、同一 seed、同一扩散日程的 **step 0** 开始：
   - 实体专家 A/B 只接收各自 standalone Short phrase；
   - 背景专家 = 同 seed 原生 SS 轨迹，只拥有所有实体 mask 的补集；
4. 每步合成：`z = M_bg ⊙ z_native + Σ M_i ⊙ z_i`，区域外只读不改；
5. 审计：`outside_write_rms≈0`、`background_native_match_rms≈0`、`partition_min/max=1`。

**已知限制**（README §5）：SS 轮廓依赖（SS 坏了 mask 就错）、固定 mask 限制形状纠正、
SAM 误差直接进路由、每步重拼接两条轨迹（交界处产生轮廓破碎/拼贴感）。

**评价协议**：最小单元 = pair × seed × 图；盲评 triptych（A-alone / B-alone / Candidate AB）
由 Qwen-VL 与 Gemini 独立评判左右实体表面风格更接近 A 还是 B（忽略形状），
正确绑定 = left→A 且 right→B；`TEST_API_KEY` 从用户环境变量读取（绝不入库）。

### EXP_1 正式结果（2026_8_25_EXP_1，100 pair × 3 seeds × 3 replicates = 900/900）
- Qwen 盲评（`experiment/2026_8_25_EXP_1/.../qwen_final_summary.json`）：
  - 漂移率：原生 SS **29.7%**（267/900）→ LSDA clean v1 **5.2%**（47/900）
  - 恢复 230 / 变坏 10 / 持续失败 37；cluster bootstrap 95% CI [-28.9%, -20.2%]
  - McNemar p≈1.7e-55；相对漂移降低 82.4%
- Gemini 已评 1800/1800（interim 快照：native 51.8% → v1 14.1%），
  **"双 VLM 均判失败才算失败"的共识终版尚未计算**（需按 eval_id 取交集）。

---

## 3. 2026-08-31 Pilot（v1 复现验证 + v2 尝试与放弃）

### 背景
为在新服务器上验证 LSDA 方法可复现，选 5 个**新 pair**（不在 EXP_1 的 100 pair 内），
seed 1011，每 pair 生成 native SS + LSDA v1 对照。期间实验负责人提出 **LSDA v2**
（"理想 LSDA"九幕设计：Scene Planner + 共享噪声 + 软核实例初始化 + Early Binding +
语义护理等），要求先做最小 pilot 原型验证可行性。

### 5 个 pilot pair
1. Dutch Delftware 青花瓶（左）vs Tunisian Qallaline 彩陶瓶（右）
2. Venetian Murano 玻璃瓶 vs Palestinian Hebron 吹制玻璃瓶
3. Chinese 青花瓷瓶 vs Thai Bencharong 彩瓷瓶
4. Japanese Bizen 备前烧陶瓶 vs French Puisaye 陶瓶
5. Spanish Toledo 嵌金盒 vs Japanese 布目象嵌金属盒

### v2 原型（已放弃）
- 实现：`Scene Planner`（弱化全局 Prompt + 固定 anchor/box）→ 三分支（场景/A/B）
  从同一噪声独立去噪前 t* 步 → 软核高斯权重一次性 Early Binding → 单共享轨迹；
  补测版加软核区域 CFG"语义护理"（t_fade=24, w_max=3.0 线性衰减）。
- Pilot 结论（1 seed × 5 pair）：**结构完整性 t*=4 最优，但色彩/纹理属性保持显著弱于
  clean v1；nursing 补测未能恢复**。经负责人决定 **2026-08-31 放弃 v2 方向**，
  正式方法保持 **LSDA clean v1**（记录见 `code/lsda/README.md` §12）。

### 保留的 pilot 数据（本地）
- `data/SS/2026_8_31_EXP_1/`（10 张 native SS，含 v1 管线同源副本）
- `data/LSDA/2026_8_31_EXP_1/`（5 张 v1 结果）
- `data/Inspection/2026_8_31_EXP_1/`（5 张 v1 comparison 三联图）
- `experiment/2026_8_31_EXP_1/`（native/v1 的 sidecars、run_configs、segmentation、archives/manifest.json）
- 服务器 `/science/wx/pry/MMDIT/` 保留全部原始副本（含 v2，供失败证据追溯）

---

## 4. 运行环境与连接纪律（重要）

- **服务器**：`ssh -p 36111 wx@s3.v100.vip`，密钥免密；工作区仅 `/science/wx/pry`
  （`models/stable-diffusion-3.5-large`、`models/sam-vit-base`、`.venv`）。
- **服务器环境**：Python 3.12.3 venv；torch 2.13.0+cu130、diffusers 0.39.0、
  transformers 5.15.1、torchvision 0.28.0（阿里云镜像安装，服务器无外网）。
- **6× A100 80GB**，小规模实验最多用 2 张；每张 v1 图约 36s、v2 约 20-40s。
- **限流纪律（2026-08-31 起）**：sshd 默认 `MaxStartups 10:30:100`，多 Agent 并发短连接
  会触发概率性丢连接。**Agent 不得高频/并行连接服务器**；需要服务器操作时
  **把命令写成清单，由实验负责人手动提交执行**；Agent 只做本地工作。
- **失败停止规则**（`warning.md` §10）：同类问题重复 3 次未解决 → 停止并汇报。

---

## 5. 当前工作区结构（2026-08-31 清理后）

```
D:\Python\MMDIT\
├── code\
│   ├── cultural100\build_cultural_pairs_100.py   # 100 文化 pair 构建
│   ├── lsda\                                      # ★ 正式方法 clean v1
│   │   ├── lsda_pipeline.py                       # 单任务流程（SAM+专家去噪）
│   │   ├── generate_lsda_900.py                   # 900 任务分片生成
│   │   ├── rate_binary_vqa_v2.py                  # 盲评 worker（openlux API）
│   │   ├── helpers\                               # phase1_common/28/212/213/221
│   │   └── README.md                              # 方法定义（含 §12 v2 放弃记录）
│   └── vqa\                                       # 盲评分析/快照/图册/watchdog
├── data\        # 只放图像：SS/LSDA/Standalone_A/B/Blind/Inspection...
├── experiment\
│   ├── 2026_8_25_EXP_1\                           # EXP_1 正式实验（900 任务全记录）
│   ├── 2026_8_31_EXP_1\                           # 本次 pilot（v1 复现，v2 已清）
│   └── README.md
├── references\  # MMDIT/SplitFlux/DreamRenderer/LayerBind 论文
├── warning.md   # ★ 存储规范与纪律（必读）
└── README.md
```

---

## 6. 下一步建议（待负责人决定）

1. **v1 pilot 扩大**：5 pair × 更多 seed（如 1011/2021/3031）→ 在服务器补跑，验证 vit-base
   SAM 下 v1 的稳定性（当前 5 pair 只有 seed 1011）；
2. **EXP_1 双 VLM 共识终版**：Qwen+Gemini 按 eval_id 取交集，计算"双 VLM 均判失败"的
   正式结论（脚本可基于 `code/vqa/analyze_qwen_final.py` 扩展）；
3. **机制分析主线**：代码尚未收录，属可复现性缺口；
4. **结构质量评价**：现有 VQA 不评形状，可补充结构指标（轮廓完整性/边缘白化等）。

---

*生成时间：2026-08-31。本文件是对当前状态的快照总结，后续实验更新时应同步修订。*
