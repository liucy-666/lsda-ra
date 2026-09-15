# code/kb — 外置文化知识库

外置文化知识库构建管线，可作为 LSDA 区域专家的知识注入源（EXP_4 / plan 9/13）。两条并存的路径：

## 路径 A：EXP_4 正式构建（openlux + `TEST_API_KEY`）
- `build_kb_fast.py`：正式构建器（enwiki 标题 + CUBE QID + batch facts，可断点续跑）。
- `enrich.py`：解析 / `verify_and_clean` / familiarity / 批处理。
- `resolve.py`：Wikidata 消歧检索（国家/非物件守卫）。
- `boost_kb.py`：native-wiki（zh/de/ko/es）补强，**未跑完**（非必须）。
- `culture_trip.py`：Culture-TRIP 相关脚本。
- 产物：`experiment/2026_9_14_EXP_4_CULTURE_KB/kb/kb_all.jsonl`（3546 条）。

## 路径 B：plan 9/13 免模型数据路径（不 import torch/diffusers）
- `parse_entities.py`：归一化 CUBE/TU 行 → `concepts_all.jsonl`。
- `build_culture_kb.py`：Wikidata 消歧 + 事实保留 + 可选 LLM 精炼；支持 `--offline` 与追加缓存。
- 用法：
  ```powershell
  cd code
  python -m kb.parse_entities --input path\cube.csv --input path\tu.jsonl --out kb\concepts_all.jsonl
  python -m kb.build_culture_kb --concepts kb\concepts_all.jsonl --offline `
    --out kb\kb_all.jsonl --coverage-report kb\coverage_report.md
  ```
  联网机器去掉 `--offline` 并提供 append-only 缓存；仅当 `OPENAI_API_KEY` 可用时加 `--llm`。

## 共享客户端
- `wikidata_client.py`：模块级函数（`search/get_entities/get_by_titles/batch_facts/collect_facts/pick_best_hit`）+ 带 JSONL 缓存的 `WikidataClient` 类。
- `llm_client.py`：openlux `chat()`（VLM 评分用，支持 `reasoning_effort`/json）+ OpenAI 兼容的 `OpenAICompatibleClient`。
- `smoke_test.py`、`watchdog_kb.ps1`、`progress_report.ps1`：冒烟 / 看门狗 / 进度。

## 约定
`TEST_API_KEY`、`OPENAI_API_KEY` 等仅从环境变量读取，禁止落盘；产物写 `experiment/<exp>/kb/`，不写入本目录。
