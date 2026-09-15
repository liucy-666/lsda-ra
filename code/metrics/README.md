# code/metrics

五个评价指标与汇总。

- VLM 双裁：由 `code/mmdit_causal/score_images.py` 完成；本目录的 `analyze_metrics5.py` / `analyze_vlm_criteria.py` 做 c1/c2/c3 汇总。
- LBP/GLCM：`metric_texture.py`。
- Gram + k-NN：`metric_dino.py`。
- SigLIP2 MaSC-CP：`score_siglip_arms.py`（`--dtype bfloat16`）、`score_siglip_census.py`。
- 汇总/图：`analyze_metrics5.py`、`analyze_objective.py`、`analyze_unified.py`、`analyze_perclass_arms.py`、`make_figures_paper.py`、`make_qualitative_gallery.py`、`make_injection_example.py`。
- 工具：`sam_masks.py`（SAM 分割）、`build_index.py`、`qc_masks.py`、`palette.py`、`arms.py`、`download_models.py`、`setup_env.ps1`、`run_eval.ps1`、`smoke_test.py`、`diagnose.py`、`print_summary.py`、`margin_stats.py`、`criterion_sensitivity.py`。

说明：CSD 指标（`score_csd.py`）及其 vendored CLIP 已移出（当前 5 指标不含 CSD）。本地内存有限（15.2GB），SigLIP 用 bf16、单进程串行。
