# MMDIT Cultural Binding

本仓库研究 SD3.5 Large 在多文化实体生成中的属性绑定问题：模型能够单独生成文化实体，却可能在组合场景中发生纹理、色彩、材料和装饰风格的跨实体泄漏。

项目包含两条主线：

- **机制分析**：追踪 MM-DiT 中 attention、FFN、gate 与残差流的跨层、跨扩散步变化，并通过因果干预检验其视觉作用。
- **方法验证**：使用 LSDA 局部专家扩散，在原生 SS 场景轮廓内为各实体分配独立 Short Prompt 专家，降低属性混合。

## 目录

- `code`：模型分析、LSDA、VQA 与实验工具代码。
- `data`：按方法和实验编号存放的图像。
- `experiment`：协议、Prompt、sidecar、盲评、统计、图表和报告。

当前实验：`experiment/2026_8_25_EXP_1`。

## 当前评价

原生 SS 与 LSDA 使用相同 seeds，并以 standalone A/B 为视觉参照。Qwen-VL 与 Gemini 分别盲评组合图左右实体更接近 A 还是 B；每张图、每个 seed 独立计分。

## 重要文档

- `code/lsda/README.md`：LSDA clean v1 的正式定义与实现边界。
- `warning.md`：目录规范、数据纪律、安全要求和禁止行为。

本项目坚持原始数据不可改写、失败样本不隐藏、假说与数据结论分离。任何新方法或新实验必须创建独立版本与实验目录。
