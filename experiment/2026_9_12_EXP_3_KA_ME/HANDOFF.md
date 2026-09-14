# HANDOFF — EXP_3：统一知识注入 + 局部重扩散（Result 2 缓解工程）

> 交接日期：2026-09-12
> 交接原因：上一 Agent 上下文过长
> **必读顺序**：`warning.md`（纪律）→ `CONVERSATION_SUMMARY.md`（项目总览）→ 本文件 →
> `experiment/2026_9_10_EXP_1_CAUSAL/HANDOFF.md`（Result 1 机制）→ `code/lsda/README.md`（LSDA 定义）
> 本实验目录：`experiment/2026_9_12_EXP_3_KA_ME/`；图像：`data/KA_ME/2026_9_12_EXP_3/`

---

## 0. 一句话现状

在 **100 cultural pairs × 3 seeds = 300 样本**上，**统一知识注入 + 局部重扩散（LSDA v1.4 矩形）** 相对原生 SS
显著降低文化错绑；**路由级 attention binding 反而更差**。5 个指标（VLM + 4 客观）已算出 2–3 臂；
**VLM 判据三选一待负责人拍板**；SigLIP 的 Binding 臂待补。

---

## 1. 贡献定位（已按负责人要求收缩）

- **方法主卖点**：**统一知识注入 + 局部重扩散**（training-free）。
  - 对每个检测到泄漏的样本：区域专家读 `实体 short prompt + 规则抽取的属性短语` 重扩散。
  - **不做 KA/ME 分流**（数据显示分流只多救 2 个样本，见 §5）。
- **分析贡献**：Result 1（内容 V 主导路由 W）+ 普查（失败主体是知识缺失 KA，占失败 76%）。
- **诊断（KA/ME）降级为分析**，不作方法机制。

---

## 2. 实验资产（本地）

### 2.1 数据

| 资产 | 路径 | 数量 |
|---|---|---|
| 普查图（A/B/SS，1024→768 jpg）| `data/KA_ME/2026_9_12_EXP_3/jpg/pair{###}_seed{seed}_{A,B,SS}.jpg` | 900 |
| LSDA 修复（ME/KA/uniform，v1.4 rect pad16）| 同目录 `lsda_{me,ka,uniform}_p*_s*.jpg` | 438 |
| Knowledge-uniform（ME+attrs）| 同目录 `lsda_meattrs_p*_s*.jpg` | 60 |
| Binding v1（**strawman，弃用**）| `data/KA_ME/2026_9_12_EXP_3/binding_jpg/` | 300 |
| **Binding v2（正式，SAM token+中层 17:26）** | `data/KA_ME/2026_9_12_EXP_3/binding2_jpg/binding2_p*_s*.jpg` | 300 |

### 2.2 评分与元数据（`experiment/2026_9_12_EXP_3_KA_ME/`）

| 文件 | 内容 |
|---|---|
| `analysis/classification.json` | 300 样本三分类（KA 190 / ME 60 / BC 50）|
| `ratings/census_scores.jsonl` | 普查 A/B/SS 双裁评分（900 行）|
| `ratings/repair_scores.jsonl` | LSDA 修复双裁（438 行）|
| `ratings/meattrs_scores.jsonl` | meattrs 双裁（60 行）|
| `ratings/binding_scores.jsonl` | **v1 strawman** 双裁（300 行，勿用于正式结果）|
| `ratings/binding2_watch/*.jsonl` | **v2 双裁（进行中，见 §6）** |
| `ratings/metric_texture.jsonl` | LBP/GLCM（899 行=3 臂）|
| `ratings/metric_dino.jsonl` | Gram + k-NN（899 行=3 臂）|
| `ratings/metric_siglip.jsonl` | SigLIP MaSC-CP（599 行=SS+LSDA；Binding 待补）|
| `manifests/` | 各评分 manifest、kb_attrs.json、repair_jobs_* |
| `figures/` | 主图/逐类/客观/消融/定性 |
| `RESULTS.md`、`PAPER_METHOD_EXP.md` | 结果报告与论文草稿（**注意：草稿里的数字是旧口径，需按 §5 更新**）|

### 2.3 代码

| 脚本 | 作用 |
|---|---|
| `code/lsda/lsda_pipeline_v14_rect.py` | **正式基座**：SAM→矩形区域→局部专家（`LSDA_RECT_PAD=16`）|
| `code/lsda/lsda_pipeline_v11_soft.py` / `v12_soft_dilate.py` | 消融（软边界/膨胀）|
| `code/lsda/run_census_lsda.py` | 批量 LSDA（读 jobs，单模型加载循环）|
| `code/lsda/run_census_binding2.py` | **正式路由基线**：SAM token + 中层 17:26 硬绑定 |
| `code/lsda/watchdog_gen.sh` | 生成看门狗（PID 文件判定，输出数达标才退出）|
| `code/mmdit_causal/gen_census.py` | 普查生成（A/B/SS）|
| `code/mmdit_causal/extract_ka_attrs.py` | 规则抽取属性短语 → `kb_attrs.json` |
| `code/mmdit_causal/make_repair_jobs.py` / `make_meattrs_jobs.py` | 生成修复 jobs |
| `code/mmdit_causal/score_images.py` | 双 VLM 评分（**必须 `reasoning_effort=none`**）|
| `code/mmdit_causal/watchdog_score.py` | 本地评分看门狗（补漏、重启、汇总）|
| `code/metrics/{arms,metric_texture,metric_dino,score_siglip_arms,analyze_metrics5}.py` | 5 指标实现与汇总 |
| `code/mmdit_causal/analyze_ka_me.py` | 旧的 5 方法分析（判据混合，谨慎）|

---

## 3. 环境与连接

- 服务器：`ssh -p 36111 wx@s3.v100.vip`（旧 A100；python `/science/wx/pry/.venv/bin/python`）
  - SD3.5：`/science/wx/pry/models/stable-diffusion-3.5-large`；SAM：`/science/wx/pry/models/sam-vit-base`
  - 代码：`/science/wx/pry/MMDIT/code/...`；实验：`/science/wx/pry/MMDIT/experiments/2026_9_12_EXP_3_KA_ME/`
  - **服务器只跑生成**；GPU0/3 曾用，现在**空闲**（45MiB/1MiB）。
  - ⚠️ SSH 会因限流被掐（本次累计 3+ 次）；按 `warning.md` §10，连续 3 次失败应停止重试并汇报。
- 本地：Windows，**RAM 仅 15.2GB**（空闲常 4–5GB）→ **大模型必须串行、单进程**。
  - 已装：torch 2.13(CPU)、transformers 5.17、scikit-image 0.26、scipy、sklearn、matplotlib。
  - 模型权重（本地）：DINOv2-large 1.16GB、CLIP-L 1.63GB、**SigLIP2-so400m 4.33GB（fp32 放不下，用 bf16≈2.2GB）**。
- API key：`TEST_API_KEY` 只从 Windows 用户环境变量读取，**禁止写入任何文件**。

---

## 4. 指标定义（5 个）

| 指标 | 做法 | 脚本 |
|---|---|---|
| **VLM** | GPT-5.4 + Gemini-3.5 双裁；判据见 §5.1 | `score_images.py` + `analyze_metrics5.py` |
| **LBP/GLCM** | 彩色三通道 LBP 直方图 + GLCM(距1/2×角4) 特征；中心裁剪 0.7；与 standalone 参考距离 → 归属 | `metric_texture.py` |
| **Gram** | DINOv2 patch 特征 Gram 矩阵 Frobenius 距离 → 归属 | `metric_dino.py` |
| **k-NN（增强原型）** | standalone 扩 5 变体（原图/翻转/3 裁剪）；区域特征对 A/B 族 top-k=5 多数投票；**不训练** | `metric_dino.py` |
| **SigLIP2 MaSC-CP** | patch 级 masked max-cosine → 归属 | `score_siglip_arms.py`（`--dtype bfloat16`）|

区域口径：左右半区；参考：同 seed standalone A/B。

### 4.1 ⚠️ VLM 判据三选一（**未定，需负责人拍板**）

| 判据 | 定义 | SS | LSDA |
|---|---|---:|---:|
| **c1 rater-strict**（项目文档口径）| 裁判任一侧<0.5 即判失败；**两裁都失败才算失败**；正确=补集 | 66.7% | **15.4%** |
| **c2 side-consensus**（analyze_deepdive 口径）| 两裁**同侧**都<0.5 才算泄漏 | 63.7% | 15.1% |
| **c3 strict-both**（保守）| 两裁**两侧**都≥0.5 才算正确 | 85.3% | 48.8% |

- c1/c2 与历史 900 任务"LSDA 恢复 90.91%"自洽；c3 与客观指标改善幅度更一致。
- 上一 Agent 的**建议**：主表 c1，脚注 c3。**未获负责人确认**。
- 注意：`RESULTS.md`/`PAPER_METHOD_EXP.md` 中的 VLM 数字（63.7→41.3）是**旧混合口径**，必须重算统一。

---

## 5. 当前结果

### 5.1 三分类（`analysis/classification.json`）

KA 190（63.3%）、ME 60（20.0%）、BC 50（16.7%）；**失败中 KA 占 76%**。

### 5.2 5 指标 × 3 臂（drift %，越低越好）

| 指标 | SS | LSDA（统一知识注入）| Binding v2 |
|---|---:|---:|---:|
| VLM（c1）| 66.7 | **15.4** | ⏳ 评分中 |
| LBP/GLCM | 80.0 | 72.2 | 83.3 |
| Gram | 74.3 | 54.8 | 79.7 |
| k-NN | 69.7 | 48.8 | 77.3 |
| SigLIP CP | 84.7 | 66.2 | ⏳ 待补 |

**结论**：① LSDA 在全部指标上改善（客观 +7.8～20.9 pts）；② **Binding 在客观指标上比 SS 更差**（路由级方法有害/无效）；③ VLM 改善幅度取决于判据（c1 最乐观）。

### 5.3 分流 vs 统一（历史，用于论证"收缩贡献"）

- LSDA uniform（全失败 short）：48.0%；Knowledge-uniform（全失败 +attrs）：41.3%；Routed（ME short/KA attrs）：40.7%。
- **差 2 个样本** → 负责人决定：方法改为统一知识注入，KA/ME 只作分析。

---

## 6. 进行中的任务（接手第一件事）

1. **Binding v2 的 VLM 双裁评分**：`ratings/binding2_watch/` 已 **~145/300**，watchdog + 2 workers 在跑。
   - 检查：`Get-Process python`；日志 `logs/watchdog_binding2.out.log`。
   - 完成后合并：把 `binding2_watch/*.jsonl`（排除 `missing/`）合并为 `ratings/binding2_scores.jsonl`。
2. **SigLIP 的 Binding 臂**（599→899）：
   ```powershell
   python code\metrics\score_siglip_arms.py --classification experiment\2026_9_12_EXP_3_KA_ME\analysis\classification.json `
     --census-dir data\KA_ME\2026_9_12_EXP_3\jpg --lsda-dir data\KA_ME\2026_9_12_EXP_3\jpg `
     --binding-dir data\KA_ME\2026_9_12_EXP_3\binding2_jpg `
     --out experiment\2026_9_12_EXP_3_KA_ME\ratings\metric_siglip.jsonl --dtype bfloat16 --arms Binding
   ```
   （脚本支持追加 + 按臂过滤；**单进程跑，勿与 DINO 并行**。）
3. **最终汇总**（Binding 齐了之后）：
   ```powershell
   python code\metrics\analyze_metrics5.py --classification ... --census-scores ...\census_scores.jsonl `
     --repair-scores ...\repair_scores.jsonl ...\meattrs_scores.jsonl --binding-scores ...\binding2_scores.jsonl `
     --texture ...\metric_texture.jsonl --dino ...\metric_dino.jsonl --siglip ...\metric_siglip.jsonl `
     --out ...\analysis\metrics5_arms.json --figure ...\figures\fig_metrics5_arms.png
   ```

---

## 7. 待办（按优先级）

1. 确认 **VLM 判据**（c1/c2/c3）→ 用统一判据重算 VLM 并与总表对齐。
2. 完成 §6 的 Binding 两臂与汇总图/表。
3. 更新论文：`PAPER_METHOD_EXP.md`（Method 改为"统一知识注入"、Metrics 五指标、主表三臂），`RESULTS.md`（数字换成统一判据）。
4. 服务器清理（可选）：binding2 驱动/看门狗已到 300/300，可停掉残留 `watchdog_gen.sh`。
5. **下一步（负责人已定，未开始）**：
   - 复现 **Culture-TRIP** 的知识文本获取 → 注入（完全体 LSDA）；
   - 外部基准：**RareBench**（R2F 仓库有提示词+GPT-4o 评测脚本+各基线结果 JSON）、**T2I-CompBench++**、**CUBE**（`github.com/google-deepmind/cube`）→ 对比已报 SOTA。
6. 候选（未做）：人工 A/B 偏好小评测（50–100 对）；线性探针（负责人已决定不做）。

---

## 8. 关键教训与红线（务必遵守）

1. **判据必须全流程一致**：分类用 c2、指标用 c3、旧脚本混合 → 数字会差 50 pts。本次踩过坑。
2. **不要用 strawman 基线**：Binding v1（颜色前景+半平面 token+全层硬屏蔽）把背景切成两半、drift 89.7%，被判定不可用；v2 才公平（SAM token + 中层 17:26）。
3. **本地内存只有 15.2GB**：SigLIP fp32 放不下（4.33GB），用 `--dtype bfloat16`；大模型**串行单进程**。
4. **SSH 限流**：连续 3 次被掐就停并汇报，不要高频重试。
5. **API key 不落盘**；VLM 评分必须 `reasoning_effort=none`。
6. **失败样本不隐藏**：2 个 `Empty latent owner`（`p073_s1012`）如实计入。
7. **图像/元数据分离**：图像在 `data/`，评分/分析/manifest 在 `experiment/`。

---

## 9. 负责人待拍板

1. **VLM 判据**：c1（项目口径）/ c3（保守）/ 两者并列？
2. 是否接受"**统一知识注入**"作为最终方法（贡献收缩）？
3. SigLIP 若仍超内存，是否允许放服务器补跑？

---

## 10. 关键文件速查

```
experiment/2026_9_12_EXP_3_KA_ME/
  HANDOFF.md              ← 本文件
  RESULTS.md              ← 结果报告（旧判据，需更新）
  PAPER_METHOD_EXP.md     ← 论文 Method/Exp 草稿（旧判据，需更新）
  analysis/classification.json, metrics5_arms.json
  ratings/                ← 所有评分（含 binding2_watch/ 进行中）
  figures/fig_metrics5_arms.png, fig_ka_me_drift.png, ...
data/KA_ME/2026_9_12_EXP_3/
  jpg/                    ← 普查 900 + LSDA 498
  binding2_jpg/           ← 正式路由基线 300
experiment/2026_9_12_EXP_2_LSDA_SOFT/CHANGE.md   ← LSDA v1.1/v1.2/v1.4 版本记录
experiment/2026_9_10_EXP_1_CAUSAL/               ← Result 1（机制）
```
