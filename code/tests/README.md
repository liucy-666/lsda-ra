# code/tests

plan 9/13 数据路径的契约测试（不加载模型权重）。

- `test_plan_contracts.py`：校验 KB 行结构、离线解析与组合 Prompt 的确定性契约。

运行：在 `code/` 下 `python -m pytest tests -q`。
