# code/culture_comp

文化组合 benchmark 与大尺度实验的编排脚本（Stage 1 / EXP_5 / EXP_6 及后续 EXP_8）。

- 概念/配对：`normalize_concepts.py`、`make_pairs.py`（跨文化 + 同类 + 稀有度；可选 `--llm-filter` 只留可绘制器物）。
- EXP_6 生图：`make_exp6_manifests.py`（LL / LSDA-Long jobs）、`gen_ll.py`（服务器，复用 native-SS 代码路径）。
- 收集：`collect_exp6.py`（旧，转 768px）、`collect_exp6_1024.py`（保留 1024）。
- 评分 manifest：`make_exp6_score_manifests.py`。
- 分析/图：`analyze_inject3.py`（三臂修复率/漂移/误伤/bootstrap）、`assemble_exp6.py`（5 指标 × 4 臂表）、`plot_exp6_permetric.py`、`make_interim_montage.py`、`make_attrs_montage.py`。
- 看门狗：`watchdog_exp6.py/.sh`、`watchdog_local_exp6.ps1`、`watchdog_metrics_exp6.ps1`、`monitor_exp6.ps1`、`run_metrics_exp6.ps1`。

约定：图像写 `data/`，manifest/评分/分析写 `experiment/`。
