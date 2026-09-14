# 3. 方法（Method）

> 论文级草稿。配图见 `figures/`。实验数字统一采用严格双裁口径（见 `Experiment.md` §4.2）。

![方法总览](figures/fig_method_pipeline.png)

**图 1.** 训练无关的缓解流程。先用 SD3.5-large 生成原生组合图；若存在属性绑定失败（严格双裁判定），则对该实体区域由"区域专家"重新扩散，其条件文本**加入规则抽取的文化属性短语**。全程不训练任何参数。

## 3.1 问题设定

研究文化实体组合生成：给定场景提示 $p_{AB}$，要求左、右分别生成文化实体 A 与 B，各自有短提示 $p_A, p_B$。当某一区域渲染成另一实体的文化表面属性（材质、色系、技法、纹样）时，称为**绑定失败**。我们假设 A、B 的单独渲染正确，只针对"组合泄漏"这一失败。目标是在**保持场景布局与背景**的前提下消除跨实体属性泄漏。

## 3.2 区域内容源：局部重新扩散

机制分析（Result 1）表明，主导误差位于注意力写回的**内容**（value $V$），而非**路由** $W$。据此，缓解应当**改写每个区域的内容**，而不动路由。

我们将其实现为**局部重新扩散（LSDA）**。在原生 SS 图上用 SAM 定位每个实体；把实例 mask 归约为其**外接矩形（pad $=16$ px）**——这使专家写入边界落在平滑背景中，避免在物体轮廓处产生硬缝。只接收实体 $i$ 提示的区域专家，对其矩形从第 $0$ 步去噪到第 $T$ 步；场景其余部分沿冻结的同 seed 原生轨迹演化。每一步 $t$：

$$
z_{t+1} \;=\; \sum_i w_i \odot \mathrm{cand}_i\!\left(z_t^{\,\mathrm{crop}},\, c_i\right)
\;+\; w_{\mathrm{bg}} \odot z_{t+1}^{\,\mathrm{native}},
\qquad w_{\mathrm{bg}} = 1-\sum_i w_i,
$$

其中 $c_i$ 为区域 $i$ 的内容源，$w_i$ 为矩形 $i$ 的抗锯齿（面积池化）软归属，$z^{\mathrm{native}}$ 为冻结的原生 SS 轨迹。每个像素始终有归属（实体专家或原生背景），边界为软混合。

## 3.3 统一知识注入

关键经验观察（见 §4.3）：多数失败源于**知识缺失**——模型连单独实体都画不出——而非跨实体污染。因此推理时**不做 KA/ME 分流**，而是对**每个被判定失败的样本**，给区域专家的内容源追加**规则抽取的属性短语**：

$$
c_i \;=\; p_i \;\|\; \mathrm{attrs}(d_i),
$$

其中 $d_i$ 为该实体的长描述，$\mathrm{attrs}(\cdot)$ 为确定性规则，抽取带属性的分句（材质、色系、技法、纹样）：

```text
attrs(d)：按 [", ", " with ", " showing ", " made of ", " made from ",
              " painted ", " decorated ", " worked ", " woven "] 切分 d
          丢弃首个名词短语；保留含属性关键词的分句
          （glaze/enamel/paint/motif/色名/porcelain/lacquer/…）
          以 "; " 拼接
```

例如
`"a Chinese blue-and-white porcelain vase with cobalt-blue underglaze painting, showing Chinese landscape, lotus, and scrolling-cloud motifs"`
得到
`"cobalt-blue underglaze painting; Chinese landscape; lotus; scrolling-cloud motifs"`。

![注入示例](figures/fig_injection_example.png)

**图 2.** 统一知识注入的工作实例（pair 081，seed 1012）。左：原生 SS 的两块地毯都被"波斯花卉"风格统一；右：LSDA 的区域专家在短提示后追加抽出的属性短语（A：羊毛绒面、中央团花、深红/蓝花卉；B：深红绒面、重复八角 gül、黑/白/暗蓝部落几何），右侧恢复为土库曼 Tekke 毯。

注入发生在**同一局部专家的内容源**上：不需要参考图、不需要适配器、不需要微调。我们刻意不做单独的 KA/ME 路由机制——在普查中，把两类分开路由只改变了**两个样本**（附录），故保持流程统一。

## 3.4 为什么是内容级，而非路由级

已有方法在**路由**层面干预（注意力 mask、token 绑定），让每个实体的 token 各自只看自己。Result 1 显示：路由只能补上表示差距的 $\mathrm{proj}_W=0.23$，而内容能补上 $\mathrm{proj}_V=0.82$（约 $3.6\times$）。仅修路由无法纠正已被污染的**内容**，这预示了结构性上限——实验上我们发现路由基线甚至有**负效果**（§4.4）。局部重新扩散则用干净、带知识的内容源替换每个区域的内容。

## 3.5 算法

```text
输入：pair (p_A, p_B)、场景 p_AB、seed s、属性短语 attrs_A, attrs_B
1. 用 SD3.5-large（seed s）生成 standalone A、standalone B 与原生 SS。
2. 诊断：是否绑定失败？（对 A/B/SS 做严格双裁）      # 仅作门控，不做分流
   若不失败：返回原生 SS。
3. 在原生 SS 上分割实体（SAM）→ 外接矩形（pad 16）。
4. 对每个实体 i：设 c_i = p_i || attrs_i。
5. 从第 0 步做局部重新扩散（软归属）；背景 = 原生 SS 轨迹。
6. 解码得到修复后的组合图。                          # 训练无关
```

诊断只用于决定**是否干预**；一旦干预，所有失败采用**相同**处理。这使方法简单且对误判稳健。
