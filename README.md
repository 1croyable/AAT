# AATField

**基于锚点的加性搬运场 (Anchor-based Additive Transport Field)**

AATField 是一种非线性分类模型。它不把样本简单看作需要被一层层矩阵变换的特征向量，而是把样本看作状态空间中的一个点，并学习一个由锚点诱导的搬运场，使样本状态在这个可学习场中逐层移动，最终完成分类。

简单地说：*样本状态 → 观察锚点并产生搬运方向与强度 → 残差移动 → 分类*

【这里建议放一张主视觉动图：二维点云在 AATField 层中逐步移动，最后被线性分类头分开】

AATField 的目标不是堆叠更大的矩阵，而是探索一种更几何化、更场化的表示演化方式：**让样本在状态空间中被逐步搬运到更容易分类的位置。**

---

## AATField 是什么？

AATField 的核心思想是：用一组可学习的 anchors 在状态空间中定义一个局部搬运场。

每一层中，样本会根据自己与 anchors 的关系，得到若干个来自 anchors 的 transport contributions。然后这些 contributions 被加总成一个移动向量，用残差形式更新样本状态：
$$
z_{next} = z_{current} + \text{transport}(z)
$$
其中：

- $z_{current}$ 是当前样本状态；
- $\text{transport}(z)$ 是由 anchors 产生的搬运向量；

多层 AATField 会让样本状态在空间中逐步演化，最后使用一个简单的线性分类头完成预测。

---

## 为什么做 AATField？

现代神经网络中，许多表示学习过程都依赖大规模矩阵变换。矩阵计算在 GPU 上非常高效，也已经被证明极其强大。

但这并不意味着矩阵变换是唯一的表示演化方式。AATField 来自一个简单的几何直觉：

> 如果分类的目标最终是让不同类别的样本在空间中变得更容易分开，那么是否可以直接学习“如何移动这些样本”，而不是只通过一层层矩阵映射去间接改变它们？

于是，AATField 尝试把分类模型看成一个可学习的搬运场：找到参考点、判断大小和方向、移动状态、重复这个过程。

这种方式使模型具有一种非常直观的解释：**每一层都在对样本状态施加一组局部的几何搬运，让样本逐渐进入更可分的空间结构**。

---

## 快速开始【还没好 先写一下pip打包】

当前项目正在整理为更标准的 Python package。
未来预期使用方式如下：

```python
import torch
from aatfield import AATField, AATFieldConfig

cfg = AATFieldConfig(
    input_dim=27,
    extra_dims=27,
    num_classes=2,
    layers=8,
    max_children=8,
)

model = AATField(cfg).to(device)

# Data-aware initialization should be called before creating the optimizer.
model.data_aware_initialize(
    train_x,
    train_y,
    samples=8192,
    min_children=2,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

logits = model(x)
```

完整训练脚本和实验配置会在 `experiments/` 中逐步整理。

---

## 核心思想

AATField 可以被概括为三个动作：观察、搬运、分类

### 1. 状态空间：在更高维中观察样本

AATField 会首先把原始输入提升到一个更高维的状态空间中。

模型首先在状态空间中放置一组可学习 anchors。这些 anchors 不是普通的隐藏神经元，而是用于定义局部几何场的参考点。样本会根据自己与 anchors 的距离关系，获得不同的响应强度。

这种设计来自一个直观假设：手中可见的数据只是更高维真实信息的低维投影。一个分类结果可能受到许多未被观测到的因素影响，而模型无法直接恢复这些隐藏因素。因此，AATField 不试图还原真实高维信息，而是**通过学习搬运场，在扩展后的状态空间中重新组织已有投影信息**，使样本逐渐变得更容易分类。

------

### 2. 产生搬运：方向、强度与激活

每个 anchor 会对样本产生一组 contribution，多个 contributions 会被加总成一个整体搬运向量。一个 contribution 同时包含：

- 方向：样本应该往哪里移动；
- 强度：这次移动有多大；
- 激活：这个 contribution 是否应该真正参与搬运。

为了让搬运场具有更强的非线性表达能力，锚点贡献上加入了方向激活。这样，搬运不再只是连续平滑地弯曲空间，而可以形成局部的折线、片状连续与边界转折。这个机制使模型能够更灵活地把类内样本聚拢，同时把类间样本拉开。

【放一张图 激活 与 不激活 对比】

------

### 3. 线性读取：让搬运后的空间变得可分

AATField 最后使用一个简单的线性分类头。这个设计为了给搬运过程提供一个明确目标：让样本经过多层搬运后，在最终状态空间中尽可能变得线性可分。

它不是把所有点压到某个固定形状上，而是**像扭动橡皮泥一样，对状态空间进行局部拉伸、折叠和重排**，使分类边界在最终空间中变得更简单。

------

## 模型结构概览【没完成 主要依靠图片】

【图片】

结构先验对于模型很重要，因为我们的模型本身是场模型 它与位置很有关系，也与结构先验时看到的数据有关系

AATField 的主线结构非常简单：

```text
Input
  ↓
State Lift
  ↓
Anchor Field Layer × N
  ↓
Linear Head
  ↓
Prediction
```

其中每个 Anchor Field Layer 完成一次：

```text
当前状态 → anchor-induced transport → 残差更新
```

当前版本的核心模块包括：

- **State Lift**：把输入样本提升到状态空间；
- **Parent Anchors**：每个类别对应的主锚点；
- **Child Anchors**：围绕主锚点生成的局部子锚点；
- **Auto-K Initialization**：根据数据自动选择每层实际使用的 child 数量；
- **Additive Transport**：将多个锚点贡献相加，形成搬运向量；
- **Directional Activation**：只保留方向上有效的局部贡献；
- **Linear Head**：对最终状态进行分类。

【这里建议放一张结构图：Input → Lift → Anchors → Transport → Residual Move → Head】

------

## 当前实验观察【放一点数据和图片】

初步实验显示：

- AATField 可以在二维几何分类任务中形成有效的非线性边界；
- 在部分小参数设置下，AATField 展示出较强的参数效率；
- 在 tabular 分类任务中，AATField 已经表现出与更大规模基线模型竞争的潜力；
- 搬运过程具有较好的可视化解释性，可以直接观察样本点如何逐层移动。

【这里建议放一个小结果表：2D / MNIST / Tabular 的参数量与准确率对比】

【这里建议放一张动态图：每一层之后样本点位置变化】

更系统的实验结果会在后续版本中整理，包括：

- 二维几何任务；
- MNIST / Fashion-MNIST；
- tabular classification；
- 参数效率对比；
- 初始化与 child 数量选择的消融实验；
- 搬运路径可视化。

------

## 许可证

本项目使用 MIT License。

你可以自由使用、修改和分发本项目代码，但需要保留原始版权声明。
