# 2026_9_15_EXP_8_DEMO_DOCES_LITCHI — SS vs LSDA 知识注入 Demo

## 目的
单对 demo，演示 "**SS=纯名词（无知识）** vs **LSDA=局部知识注入**"。用负责人指定的一对：Brazil Doces do Brasil（左）× China litchi（右）。

## Prompt（冻结）
- **SS（纯名词，全局）**：`Neutral studio background: Doces do Brasil on the left, litchi on the right; both fully visible, separate, and similar in size.`
- **LSDA-A（区域专家，知识）**：`Doces do Brasil, vibrant colors, sugar-based, small bite-sized forms, intricate shapes, decorated with sprinkles or coconut, festive appearance`
- **LSDA-B（区域专家，知识）**：`litchi, red or pink skin, translucent white flesh, sweet and juicy, round form, glossy surface`

## 配置
- 变体：**v15mask**（SAM 轮廓 + dilate24 + feather12；矩形仅算 crop）。
- SD3.5-large，1024，28 steps，CFG 4.5，fp16；seeds 42/43/44。
- 运行：`code/lsda/run_census_lsda_v15.py --variant mask --dilate-px 24 --feather-px 12 --rect-pad 16`（内部同时产出 `native_ss.png` 与 `lsda.png`，同 seed 可比）。
- 图：`data/Demo_Doces_Litchi/2026_9_15_EXP_8/`。

## 进度
- [x] 生成 3 seeds（SS + LSDA，v15mask）
- [x] 生成 standalone A/B（知识版单实体，seed 42）
- [x] 并排对比图 + 目视报告

## 结果（`figures/fig_demo.png`，`figures/fig_demo_best.png`=构图最好的 s42）
列 = standalone A（知识）/ standalone B（知识）/ SS（纯名词）/ LSDA（v15mask）。

- **standalone A（知识）**：正确生成巴西甜点（彩色小糖果、brigadeiro/beijinho 类）。
- **standalone B（知识）**：正确生成荔枝（红/粉、鳞状表皮）。
- **SS（纯名词 "Doces do Brasil" + "litchi"）**：**失败**——只知道是"红色圆果"，没有画巴西甜点（KA：知识缺失），两侧都成红色圆果实。
- **LSDA（知识注入）**：左侧成功出现**彩色巴西甜点**、右侧为荔枝 → **局部知识注入生效**（无知识→有知识）。
- **代价/限制**：LSDA 只能写入 SS 分割出的区域；因 SS 把两位都画成圆果，左区被注入甜点后与右侧大果**重叠/拥挤**，构图不佳（s42 构图最好，贴边=0）。

## 结论
1. 纯名词 SS 对该长尾文化食物**知识缺失**，无法正确生成；这是 KA 的直接演示。
2. standalone 知识提示可正确生成两端；**LSDA 把外部知识注入局部专家后，同样能补回缺失知识**（左侧甜点出现）。
3. LSDA 输出受**原生 SS 构图/分割**制约：SS 布局差 → 注入后重叠。→ 与 EXP_7 结论一致：需先保证 SS 构图（小尺寸/留白/选种）。

## 文件
- 图源：`data/Demo_Doces_Litchi/2026_9_15_EXP_8/{SA,SB,s42,s43,s44}/`。
- 脚本：`code/lsda/run_census_lsda_v15.py`（LSDA）、`code/culture_comp/gen_pairs_census.py`（standalone）、`code/culture_comp/make_demo_montage.py`（对比图）。
