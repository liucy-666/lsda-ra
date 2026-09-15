# code/mmdit_causal

受控 100 对实验的核心生成/评分/分析代码（EXP_3 / EXP_6）。

- 数据：`cultural_pairs_100.json`（100 对，含 short/long/SS/SL/LL）、`pairs100.py` / `pairs.py`（提示词构造）。
- 生成（服务器）：`gen_census.py`（A/B/SS）、`gen_deepdive.py`、`mmdit_lib.py`（内部激活/hook）。
- 评分：`score_images.py`（双 VLM，**必须 `reasoning_effort=none`**）、`ask_vlm.py`、`watchdog_score.py`、`audit_figures.py`（美学审计）。
- 知识/属性：`extract_ka_attrs.py`（规则抽取属性短语）、`classify_census.py`（KA/ME/BC 三分类）。
- manifest：`build_*_manifest.py`、`make_repair_jobs.py`、`make_meattrs_jobs.py`、`select_seeds.py`、`split_manifest.py`。
- 机制分析：`analyze_axis.py`、`analyze_e2.py`、`analyze_deepdive.py`、`run_e1.py`、`run_e2.py`、`run_internal.py`、`run_axis.py`、`run_deepdive_*`。
- 图：`make_fig_*.py`、`make_schematic.py`、`make_demo_montage.py`、`make_sidebyside.py`。

约定：`TEST_API_KEY` 只从环境变量读；图像在 `data/`，评分/分析在 `experiment/`。
