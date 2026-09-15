# 2026_9_14_EXP_4_CULTURE_KB — 外置文化知识库（Stage 1 / Task 2）

## 目的
为 CUBE-1K + TU gold_concepts 的 3546 个文化概念构建**外置知识库**，作为 LSDA 区域专家的知识注入源；可独立成库发布。

## 流程
归一化 → GPT-4o-mini 解析 `{culture, object_type, style}` → Wikidata 消歧检索（国家/非物件守卫）→ facts 抽取 + 1-hop 标签 → facts 拼接 + LLM 去冗余 = `knowledge_text`；检索不到则 gpt-4o 生成（`source=llm_fallback`）。`verify_and_clean` 拦截错误实体（如 yugwa→印尼村庄、niu jiao hu→篮球队）。

## 产物与结论
- `kb/kb_all.jsonl`：3546 条，100% hit，**0 error**；`source=wikidata 464 (13.1%)`、`llm_fallback 3082 (86.9%)`。
- `protocol/SMOKE_TEST.json`：3/3 冒烟通过（mini 不支持 reasoning_effort；JSON 模式需含 "json"）。
- 字段含 `facts / knowledge_text / familiarity / source`。
- **Wiki 率 13.1% 未达协议硬指标 ≥20%**（Wikipedia/Google 在本环境被墙），如实记为 protocol deviation，HANDOFF 判定「Wiki 率不设硬指标」。

## 进度与未来
KB 构建完成。未来方向已变更为：为「可视化文化实体」重构 KB，并新增**结构化视觉属性**（material/palette/motif/technique/form）供 LSDA-Attrs 注入（见计划 EXP_7）；`boost_kb.py`（native-wiki 补强）未跑完，非必须。
