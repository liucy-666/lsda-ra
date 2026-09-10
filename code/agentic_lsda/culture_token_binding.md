# 文化场景下的注意力绑定：DreamRenderer 不足 + Token Binding 改进

> 目标：回答"DreamRenderer 的方法在做**多实例文化绑定**（如中青花 vs 意美奥利卡，共享蓝白、低判别度）时
> 哪儿不够用"，再据此提出"文化感知 Token Binding"改进，并在 SD3.5 上验证。

## 0. 我们的文化场景（与论文 layout-to-image 的区别对照）

| 维度 | DreamRenderer | 我们的文化绑定 |
|---|---|---|
| 任务 | layout-to-image（depth/canny + box/mask 定位实例） | 纯 text-to-image（一个 prompt，无 box/mask） |
| 实例判别 | 实例是显式对象，边界由用户给定 | 实例是"同一类物体（花瓶）×不同文化"，边界需自动求 |
| 失败模式 | 属性错误（如颜色/形状绑错） | **文化风格串扰/漏配**：A 的文化元素出现在 B 上，或二者混为一体 |
| 判别难度 | 属性通常彼此不同 | 两实例**共享蓝白配色**，仅纹样/构图/文化语义不同 → 低判别度 |

DreamRenderer 的核心（§3.5/§3.6）：**用注意力 mask 把每个实例的词元"关进"自己的子空间**——图像 token 只 attend
自身图像+自身文本（`M^hard`），文本 token 只 attend 自身+桥接图像 token（`M^tiext`）；并把 hard 只放在 **vital 层**
（FLUX 57 层 Joint Attention 的中间层），其余层用 soft（图像 token 允许看整图 `M^soft`）以保全局和谐。

## 1. DreamRenderer 在文化场景的不足

**① 依赖布局输入（最大硬伤）。** 论文要求用户给 depth/canny 结构图 + 每实例 box/mask，才能知道"哪些图像 token 属于实例
i"。纯 T2I 文化绑定没有这些，必须自动分割实例。分割一旦出错，绑定就绑到了错误区域——**错误的绑定比不绑更糟**（会把
A 的图像 token 错绑给 B 的 mask）。我当前用颜色阈值分割（把整帧判成前景 A=512/B=512），只能说"粗粒度可用"。

**② "类别"与"文化属性"混淆。** `M^hard` 绑定的是"实例 i 的所有图像 token ↔ 实例 i 的所有文本 token"。但当两个实例
共享同一**名词类**（都是"vase"）时，这只能锁住"vase 属于哪个实例"，**锁不住"哪种文化风格属于哪个实例"**。文化是修饰
属性（Chinese / maiolica / landscape / floral），它需要**属性词元→实例**的绑定，而不是整个实例文本段的绑定。

**③ 共享修饰词无法判别（低判别度的根源）。** 两实例都叫"blue-and-white vase"。`M^tiext` 把每个实例的文本隔离，
但如果"blue-and-white"这类**共享描述词**同时出现在两个实例的描述里，硬隔离并不会把它们正确分派——它们天然可同时属于
两者。DreamRenderer 假设"每个实例文本彼此不同"，文化场景恰好**不满足**这一点。

**④ soft 层正是文化泄漏的通道。** `M^soft` 允许实例图像 token attend **整张图**。为了全局和谐，DreamRenderer 在
非 vital 层放松隔离。但文化场景里"串风格"恰恰是**最主要**的失败模式：实例图像 token 能看到全图，就会在 soft 层把
另一实例的文化纹理/纹样"看进来"。DreamRenderer 为保质量而接受的泄漏，在文化绑定里就是核心缺陷。

**⑤ 跨步静态、不感知时机。** 绑定在每一步都用同一套 mask。我们的 LSDA 时序分析表明文化漂移/混淆往往在**特定时间步**
涌现（越早干预越有效）。静态绑定无法在漂移发生时加强、在稳定时放松。

**⑥ 桥接图像 token 的代价。** `M^tiext` 需为每个实例复制一份图像 token（不参与输出，仅辅助文本绑定），增加计算，
也进一步依赖"哪些 token 是哪个实例"的掩码。

## 2. 改进：文化感知 Token Binding（Culture-aware Token Binding）

在 DreamRenderer 的"实例级隔离"之上，加一层**"文化属性词元→实例"的绑定**，并针对上述不足逐条修正：

**(A) 文化属性词元绑定（针对 ②③）**
把 prompt 结构化：`[global] [A: desc_A_culture] [B: desc_B_culture]`，tokenize 后把 **A 独有的文化词元**（如 Chinese、
landscape、pagoda、mountain）绑到 A 的实例区域，把 **B 独有的**（maiolica、floral、arabesque）绑到 B 实例区域。
- 绑定粒度从"整段文本"降到"文化语义词元"；
- 用 `M^tiext`-style：A 的文化词元只 attend A 的桥接图像 token + 自身。
**(B) 共享词元用 soft 处理。** 对确实两实例共享的修饰词（blue-white），**不强绑**，用 soft 让其参与全局——因为它本就属于
两者，强绑会引入错误先验。
**(C) 自动实例分割（针对 ①）**。对前景做连通域，取最大两块作为 A、B，其余为背景 `I_bg`；对每个实例得到 `I_A`、`I_B`，
并在绑定时把背景 token 归到"任意可见"（soft）。这样 A/B 不重叠、且把背景排除在 hard 之外（避免我当前"整帧都是前景"。
**(D) hard/soft 分层 + critical 层搜索（针对 ④）**。用 A1-1 的注意力 hook 找出 SD3.5 的 vital 层：hard 只放这些层，
其余层用 soft。**关键创新**：把"soft 允许看全图"改成"soft 允许看全图，但**对另一实例的文化词元/区域打一个衰减**"，在保
和谐的同时堵住文化泄漏。
**(E) 时机感知绑定（针对 ⑤，与 LSDA 集成）**。用 v6 诊断器判断每步漂移程度，漂移高的步加强绑定强度（hard 层更多/更硬），
稳定步放松。这是"Token Binding + 时机"的联合，规避了我之前分支打分中"时机间隔偏弱"的问题——把时机信号用在绑定强度
上，而不是用在"重roll"上。

## 3. 验证协议（SD3.5）

1. 生成若干 seed 的**原生**文化对，筛出"原生已文化混淆/漏配"的**失败用例**；
2. 对同一 prompt/seed 应用改进绑定（A+B+D），生成绑定图；
3. 三路对照：**原生 vs DreamRenderer 式（图像 hard 全层） vs 文化感知 Token Binding（属性词元绑定 + vital/soft）**；
4. 用像素 diff + 视觉核查 + （若装有 Pwen-VL-3B+lora_v6）v6 诊断器打分，判定改进版是否把失败用例"救回"正确归属。

## 4. 简化先做（本轮最小可行验证）

先落地 **C(自动分割)→强化分割、D(vital 层 hard + 其余 soft)、B(共享词元 soft)** 三者，与已有 A↔B 屏蔽组合，
在"原生已混淆"的用例上验证改进版能否救回。属性词元绑定(②③)与时机感知(⑤)作为下一阶段。

*本文档归属 `code/agentic_lsda/`，对应 CONVERSATION_SUMMARY.md §11。*
