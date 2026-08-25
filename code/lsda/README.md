# LSDA：本项目中的方法定义与实现边界

> 本文档用于防止后续开发者把 LSDA 误解成普通局部重绘、图像粘贴、SL/LL donor 替换或仅在扩散后期进行的 mask 修补。

## 1. LSDA 要解决什么问题

LSDA 面向以下情形：模型能够用单实体 Short Prompt 生成文化实体 A 和 B，但在多实体 Short–Short（SS）组合中，一个实体的文化纹理、色彩体系、材料观感或装饰纹样被另一实体污染。

LSDA 的目标不是通过更长的组合 Prompt 掩盖问题，而是在保留 SS 场景布局的同时，让每个实体区域由只接收该实体 Short Prompt 的局部专家独立去噪，从而降低跨实体属性泄漏。

当前项目的 LSDA 成功标准应与 standalone controls 比较：组合图中左、右实体的文化表面特征分别更接近对应的 A-alone、B-alone，而不是互相串用。当前大规模 VQA 只评价材料观感、颜色、纹理、纹样、装饰技术和表面处理；形状相似度应作为独立结构指标报告，不能混入纹理绑定得分。

## 2. 当前正式实现：LSDA clean v1

权威入口：

- `lsda_pipeline.py`：单任务完整流程与局部专家去噪核心。
- `generate_lsda_900.py`：900 个冻结任务的分片生成入口。

单个任务的数据流如下：

1. 使用组合 SS Prompt 和给定 seed 生成原生 SS 去噪轨迹及最终 SS 图像。
2. 在原生 SS 图像上使用 SAM 获得各实体的实例轮廓。
3. 将重叠轮廓仲裁为严格互斥的 owner masks，并建立背景补集。每个图像像素及对应 latent cell 只能有一个 owner。
4. 所有专家从同一初始 latent、同一 seed、同一扩散日程的 step 0 开始运行：
   - 实体专家 A 只接收 standalone A 的 Short phrase；
   - 实体专家 B 只接收 standalone B 的 Short phrase；
   - 第 N+1 个背景专家使用同 seed 的原生 SS 轨迹，只拥有所有实体 mask 的补集。
5. 每个实体专家可以读取其包围 crop 的上下文，但只有其 owner mask 内的 candidate latent 会被提交；mask 外的该专家写入恒为零。
6. 每一步将 N 个实体专家的互斥写入与背景 SS 状态合成为下一 latent，最后由同一 VAE 解码。

形式化地，在扩散步 `t`，合成状态为：

```text
z_(t+1) = M_bg ⊙ z_native_(t+1)
          + Σ_i M_i ⊙ z_i,(t+1)
```

其中 `M_bg + Σ_i M_i = 1`，且任意两个实体 mask 不重叠。`z_i,(t+1)` 是实体 i 的 Short Prompt 局部专家从当前共享状态计算的候选更新。

因此，“区域外只读不改”的准确含义是：某个实体专家不能向自己的 mask 外提交写入；区域外仍由其他实体专家或背景 SS 轨迹合法更新，并不是冻结为某一张静态图像的像素。

## 3. 允许和禁止使用的信息

### 生成阶段允许

- 原生 SS Prompt、seed 和完整去噪轨迹；
- 从同 seed 原生 SS 图像获得的 SAM 实例轮廓；
- 每个实体的 standalone **Short phrase**；
- SS mask 补集上的背景轨迹。

### 生成阶段禁止

- SL 或 LL 图像、mask、latent、hidden state 作为 donor；
- standalone A/B 图像、latent 或 hidden state 作为 donor；
- 用 standalone 图像直接粘贴、融合或初始化实体区域；
- 给局部专家输入组合长文本或另一实体的描述；
- 一个 latent cell 同时接受两个实体专家写入；
- 把原生 SS 最终图像做一次普通 inpainting 后称为 LSDA。

Standalone A/B 图像只用于构建知识充足性基线和最终评价，不参与 LSDA 生成。

## 4. LSDA 不是什么

- **不是后处理贴图。** 没有把 standalone 图像复制进组合场景。
- **不是单次 mask inpainting。** 局部专家从 step 0 参与全部去噪步。
- **不是 SL/LL rescue。** 当前生成路径不读取 SL/LL 的任何视觉或内部状态。
- **不是矩形区域独占。** crop 仅用于计算效率与上下文读取，真正的写权限由 SAM one-hot mask 决定。
- **不是完全独立的 N 张图再拼接。** 专家共享当前组合 latent 的上下文和扩散时间，但其提交写入彼此互斥。
- **不是已经证明的通用解决方案。** 是否恢复文化属性必须由冻结 seed 的 SS–LSDA 配对数据和盲评结果决定。

## 5. 当前实现的关键限制

1. **SS 轮廓依赖。** 当前 owner masks 来源于原生 SS。因此，若 SS 已经漏掉实体、严重融合或给出错误轮廓，局部专家没有可靠区域可接管。
2. **形状纠正能力受限。** 固定 owner mask 天然偏向保留 SS 的几何边界；局部专家可以重写区域内部，但不能自由消除旧轮廓或在 mask 外建立大幅不同的新形状。纹理恢复成功不等于物理形状恢复成功。
3. **SAM 误差会直接进入路由。** 遮挡、叠放、接触和细长结构可能造成错误分割或重叠仲裁。
4. **latent 下采样会损失细边界。** 图像空间 one-hot mask 在缩放到 latent 网格时可能丢失很小的部件；代码会检查最终 latent ownership 是否仍严格 one-hot。
5. **共享上下文仍可能传递偏置。** 写权限互斥不代表专家计算完全隔离；局部 crop 中读取的共享状态仍包含全局场景信息。

这些限制必须作为结果的一部分报告，不能把失败样本删除或解释掉。

## 6. 必须保存的审计字段

每张正式 LSDA 图像必须有对应 sidecar，并至少保存：

- task/pair ID、seed、初始 latent SHA-256；
- 全局 SS Prompt 与实体 Short Prompts；
- SAM mask 来源和 owner 面积；
- image-space 与 latent-space one-hot 检查；
- 每一步 `outside_write_rms`；
- 每一步 `background_native_match_rms`；
- 扩散 steps、CFG、模型版本和运行时长；
- 明确记录 standalone/SL/LL image、latent、hidden state 均未作为 donor。

理论上应满足：

```text
outside_write_rms ≈ 0
background_native_match_rms ≈ 0
partition_min = partition_max = 1
```

若这些条件不成立，该样本不能被称为 strict one-hot LSDA。

## 7. 评价口径

最小评价单元是“一个 pair × 一个 seed × 一张图”，不得把三个 seeds 合并成一次成功或失败。

当前纹理归属 VQA 同时展示匿名化的 A-alone、B-alone 和 Candidate AB，只问：

1. Candidate AB 右侧实体的表面文化特征更接近 A-alone 还是 B-alone？
2. Candidate AB 左侧实体的表面文化特征更接近 A-alone 还是 B-alone？

正确绑定为 `left → A` 且 `right → B`。原生 SS 与 LSDA 必须使用同一批 seeds、同一 standalone controls 和同一盲评规则，最后报告成功图像数、失败图像数及逐 seed 清单。

知识不足与组合绑定失败必须分开：只有 A-alone、B-alone 均正确而原生 SS 失败的样本，才能用于评价 LSDA 是否修复了“模型有知识但组合绑定失败”的问题。

## 8. 版本与命名

- `LSDA clean v1`：本文档定义的严格 one-hot、step-0 局部专家版本，是当前正式实现。
- 旧的 `LSDA-X` / `LSDA-RA` 名称来自探索阶段，不应与 clean v1 混用；除非其数据流满足本文全部约束，否则不得汇入正式 LSDA 统计。
- `native SS`：未经局部专家处理、与 LSDA 使用相同 seed 的组合短 Prompt 基线。

若未来修改 mask 来源、专家启动步、prompt 来源、背景专家或合成公式，必须创建新版本号和新实验目录，不能静默覆盖 clean v1。

## 9. 路径约定

- 代码：`D:\Python\MMDIT\code\lsda`
- 图像：`D:\Python\MMDIT\data\<method>\<experiment_id>`
- 配置、sidecars、日志、评分及分析：`D:\Python\MMDIT\experiment\<experiment_id>`

不要把图像、JSON、评分或日志重新放回本目录。

## 10. 运行时路径配置

仓库不再绑定任何个人工作目录或特定服务器目录。模型权重和辅助模块体积较大，
不随仓库分发，运行时必须显式传入，避免把某台机器的目录误认为项目约定。

单任务入口 `lsda_pipeline.py` 必须指定：

- `--model-dir`：本机或计算节点上的 SD3.5 Large 权重目录；
- `--sam-model-dir`：SAM 权重目录；
- `--helpers-dir`：包含 phase1/phase2 模块的 LSDA 辅助代码目录；
- `--output-root`：本次运行的独立输出根目录。

900 任务入口 `generate_lsda_900.py` 必须另外指定冻结 manifest、原生 SS 图像根目录、
对应的 SAM mask 根目录和输出目录。mask 可以来自早期分割任务，但这里只将其作为同 seed
原生 SS 的分割产物读取；旧 LSDA/RA 图像、latent 或 hidden state 不会被读取。

`helpers-dir` 中的辅助模块目前尚未收录进本仓库，这是完整复现前必须补齐并锁定版本的外部依赖。

## 11. 历史代码清理记录

`code/cultural100/experiment_4500` 曾保存 Base、LSDA-RA、Original 与早期 VLM 的一次性脚本。
这些脚本依赖已经废弃的目录结构，且不属于 LSDA clean v1 正式实现，已从代码区移除。
其生成图像、原始评分、日志与实验元数据仍完整保留在 `data` 和 `experiment` 中，
不得据此将历史探索结果并入 clean v1 的正式统计。
