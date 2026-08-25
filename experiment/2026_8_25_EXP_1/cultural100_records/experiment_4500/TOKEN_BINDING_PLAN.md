# Token Binding 对照实验计划（待实现冻结）

## 目标

在完全相同的 100 个文化 Pair、SS Prompt、9 个 latent 和 SD3.5 Large 配置下，比较 Native SS、LSDA-RA 与 Token Binding。Token Binding 不读取 standalone 图像，以避免与 LSDA 的视觉锚点信息混淆。

## 复现前必须冻结

1. 论文、官方仓库和 commit hash；若官方实现不支持 SD3.5/MM-DiT，必须明确记录移植层。
2. 需要干预的文本编码器、token span、层、时间步和强度参数。
3. 多 token 文化实体的聚合规则，例如 `Chinese blue-and-white porcelain plate` 不能只绑定单个词。
4. 是否需要空间 mask；若需要，mask 必须来自 Native SS 本身或固定布局先验，不能读取 LSDA 或 SL/LL 输出。
5. 参数选择只允许在独立 pilot Pair 上完成一次，随后冻结用于全部 100 Pair。

## 生成矩阵

- 100 Pair × 3 seed groups × 3 replicates = 900 张 Token-Binding SS。
- 每张图复用 Native SS 对应的 `latent_seed`、Prompt、28 steps、CFG 4.5 和 1024×1024 分辨率。
- 输出逐图 sidecar：token spans、干预位置、强度、运行时间、峰值显存和代码版本。

## Pilot 与参数冻结

- 从 100 Pair 中按编号预先选择 5 Pair，不按生成效果挑选。
- 在每个 Pair 的第一个 latent 上比较不超过 4 组参数，共不超过 20 张 pilot 图。
- 依据双实体纹理保真度、属性泄漏和结构完整性选择一组全局参数；不为每个 Pair 单独调参。

## 盲评终点

- 每个实体的材质、颜色、工艺、纹样得分及其均值 `T_A/T_B`。
- 主连续指标 `BTF=min(T_A,T_B)`。
- 主成功率 `DualTexture@4`。
- `A→B` 与 `B→A` 属性泄漏。
- 轮廓相似度 `S_A/S_B`、双实体存在、融合、额外物体和重影。
- Native、LSDA、Token Binding 以匿名方法编号交给 Qwen-VL 与 Gemini 独立评审。

## 统计比较

- 同一 Pair、同一 latent 的配对差：Token Binding−Native、LSDA−Native、LSDA−Token Binding。
- 先报告每个 Pair 的效果，再做 Pair 等权的宏平均和 bootstrap 置信区间。
- 二分类成功率使用配对比例差；连续指标使用 seed-paired 差值。

## 公平性边界

- Token Binding 不使用 standalone 视觉参考；LSDA 使用 standalone Short 视觉锚点，因此二者代表不同信息预算。
- 论文中同时报告“方法性能”和“额外信息来源”，不能把二者描述为完全同预算算法。
