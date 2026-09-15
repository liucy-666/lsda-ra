# 2026_9_14_EXP_5_CULTURE_COMP — 文化组合 Benchmark（Stage 1 / Task 1）

## 目的
在受控 100 对之外，用 CUBE-1K + TU gold_concepts 构造 **1000 个组合文化 Pair**（seed 42），生图 SS / LSDA 并评测漂移与副作用。分 4 批（250/批），批间审批。

## 产物
- `benchmark/concepts_all.jsonl`：3546 条（CUBE 1002 + TU 2544；物件型 2675 / 唯一 2551）。
- `benchmark/cube_1k.json`、`gold_concepts.csv`、`gold_concepts_translated.csv`。
- `benchmark/pilot_pairs_3per.jsonl`：15 对 pilot（每类 3 对）。
- 配对规则：跨文化 + 同类；稀有度一侧 ≤0.4、另一侧 ≥0.6；模板「中立背景，A 左 B 右，均完整且大小相近」。

## Pilot 结论（关键）
- **CUBE/TU 缺乏同材质/同器型的跨文化配对**，直接同类配对会得到不可比对象（如 peas × saeukkang）。
- art/music 类混入**表演/舞蹈/节庆**；food 细节泛化、视觉区分度低；类别存在错标（kurta 归 art 等）。
- **建议改用 Wikidata 反查构造**（材质 P186 + 器型 P31/P279 + 国家 + 图 P18），而非只用 CUBE/TU。

## 进度与未来
Stage 1 采样/pilot 完成，**未进入批量生图**。未来：先做「可视化实体选取 + KB」再重启本 benchmark（配额建议 clothing/utensil/instrument 为主）。
