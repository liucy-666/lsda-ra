# code/kb

外置文化知识库构建管线（EXP_4），可作为 LSDA 知识注入源。

- `build_kb_fast.py`：当前正式构建器（enwiki 标题 + CUBE QID + batch facts，可断点续跑）。
- `enrich.py`：解析 / `verify_and_clean` / familiarity / 批处理。
- `resolve.py`：Wikidata 消歧检索（国家/非物件守卫）。
- `wikidata_client.py` / `llm_client.py`：Wikidata API 与 openlux LLM 客户端。
- `build_kb.py`、`build_culture_kb.py`：早期版本（已被 `build_kb_fast.py` 取代）。
- `boost_kb.py`：native-wiki（zh/de/ko/es）补强，**未跑完**（非必须）。
- `culture_trip.py`：Culture-TRIP 相关脚本。
- `smoke_test.py`、`watchdog_kb.ps1`、`progress_report.ps1`：冒烟/看门狗/进度。

约定：`TEST_API_KEY` 只从环境变量读，禁止落盘；产物写 `experiment/<exp>/kb/`。
