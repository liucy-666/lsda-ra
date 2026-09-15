# code/benchmark

plan 9/13 的免模型跨文化 prompt 组合工具，从 KB 生成确定性的组合 Pair。

- `build_composed_prompts.py`：把概念/知识合成为跨文化组合 Prompt。
- `convert_to_pairs.py`：转换为实验可用的 pair/manifest 结构。

不依赖 torch / diffusers / 模型权重；产物写 `experiment/<exp>/`，不写入本目录。
