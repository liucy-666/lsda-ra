# MMDIT experiment layout

- `D:\Python\MMDIT\code`: source code and launch scripts only.
- `D:\Python\MMDIT\data`: image files only, grouped by generation method and experiment ID.
- `D:\Python\MMDIT\experiment`: manifests, prompts, sidecars, ratings, logs, analyses, and source data.

Each new experiment uses a folder named `YYYY_M_D_EXP_N`, for example `2026_8_25_EXP_1`.

The active blind-review watchdog status is stored under:

`2026_8_25_EXP_1/cultural100_records/experiment_4500/binary_vqa_v2/watchdog/status.json`
