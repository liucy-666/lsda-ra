# code/vqa

900 任务（2026_8_25_EXP_1）的 VQA 评分与报告脚本。

- `analyze_qwen_final.py`：Qwen 1800/1800 盲评终版报告（1800 张；结果见 `experiment/2026_8_25_EXP_1/README.md`）。
- `compute_binary_vqa_snapshot.py`：二分类 VQA 快照。
- `export_qwen_worsened_gallery.py`：导出「原生正确→LSDA 变坏」图集（副作用证据，必须保留）。
- `watch_vqa_v2.ps1`：评分看门狗。

说明：主判据已转为双 VLM（Qwen+Gemini）交集，本目录 Qwen 结果作为单裁证据与历史。
