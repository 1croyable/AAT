# Auto-K Initialization Research

本实验用于研究 AATField 中 child anchor 数量的选择机制。当前 v0.1 版本使用 layer-local Auto-K 方法：每一层根据当前状态空间中的样本分布，为该层选择一个共享的 child 数量。然而，从已有实验结果看，不同任务、不同层数和不同 state 维度下，child 数量会显著影响模型表现，而且单层最优的 K 不一定代表整条 transport stack 的最优配置。因此，本实验希望通过固定 child 数量、枚举不同层间 K 组合，并与当前 Auto-K 结果进行对比，观察 child 数量在不同层中的作用规律，进一步探索是否可以设计出更稳定、更全局的初始化选择算法。

This experiment investigates the selection mechanism for the number of child anchors in AATField. The current v0.1 implementation uses a layer-local Auto-K method: each layer selects a shared child count based on the sample distribution in the current state space. However, existing experimental results suggest that the number of children can significantly affect model performance across different tasks, layer depths, and state dimensions. Moreover, the best K for a single layer may not correspond to the best configuration for the entire transport stack. Therefore, this experiment aims to compare fixed child counts, enumerate different layer-wise K combinations, and benchmark them against the current Auto-K strategy, in order to observe how child count affects different layers and explore whether a more stable and more global initialization selection algorithm can be designed.

---

## Notes

- 3d棋盘实验表明，当前 Auto-K 的评分标准严重低估了 checkerboard 这种多局部区域任务需要的 child 数量。K≥34以后就很好了但是他选了5。

- 3d棋盘实验中：
  K=2~8      很弱
  K=9~20     缓慢上升，但不稳定
  K=21~33    明显爬坡
  K=34以后   进入 0.95 左右的平台区
  K=49~80    基本平台震荡，没有明显继续提升
  ![image-20260617214936403](C:\Users\jerem\AppData\Roaming\Typora\typora-user-images\image-20260617214936403.png)

- 之前我以为scailing是依靠child数，后又经过小实验判断是维度数，但现在看来不一定，child增大会波动但是总体还是在一个高位。而维度数也不一定不是，需要在autoK实验做完后继续做相关实验，如选择K=35，尝试改变维度数，或者看在不同维度数下性能最好的K，这需要交叉实验，但是交叉内容太多，需要挑选合适区间比如K=30~50，维度数若干。

- benchmark中，MLP-large达到91%，较小的MLP无法学到好的情况。说明AAT在这种几何结构明显的任务中拥有更强的归纳偏置，参数效率更高。

- Fisher score 在 K≈25~34 附近快速跃迁，然后在高位平台。这说明准确率提升不是偶然的，确实是 transport 后的状态变得更可分了。

  ![image-20260617215709523](C:\Users\jerem\AppData\Roaming\Typora\typora-user-images\image-20260617215709523.png)

- K 太小时，场没有足够的局部支撑点，搬运后类别仍混在一起。K 到达 30+ 后，场终于能把 checkerboard 的局部交错结构展开。

- 高 K 之后不是所有 child 都在强烈激活，搬运距离响应变化不大。child 更多了，但每个点只需要更局部、更稀疏地使用其中一部分 child。合理。

- ![image-20260617215818329](C:\Users\jerem\AppData\Roaming\Typora\typora-user-images\image-20260617215818329.png)

- 经过实验
  ```
  === selected K by method ===
  method                old_arg  hard95  hard_knee  hard_arg  soft95  resp95  purity95  inertia_knee
  ---------------------------------------------------------------------------------------------------------
  weighted_kmeans            5      79         32        80      78      74        46            17
  unweighted_kmeans          5      74         32        80      78      68        46            17
  farthest_boundary         25      30         25        33      25      66        13            21
  kmeanspp                  17      72         78        72      72      73        49            13
  pca_split                 16      76         59        78      76      79        47            17
  
  
  === selector stability ===
  selector              mean     std    min    max   in_34_60
  --------------------------------------------------------------
  old_fisher_argmax     29.40   29.16      5     69        1/5
  hard_nmi_knee         33.60    0.89     33     35        2/5
  hard_nmi_95           33.60    0.89     33     35        2/5
  purity_90             30.00    0.00     30     30        0/5
  purity_92             30.20    0.45     30     31        0/5
  purity_95             31.80    0.84     31     33        0/5
  purity_98             32.80    0.84     32     34        1/5
  inertia_knee          15.20    1.92     12     17        0/5
  hard_nmi_argmax       33.60    0.89     33     35        2/5
  purity_argmax         33.60    0.89     33     35        2/5
  ```

- 决定使用 kmeans聚类 + hard_nmi_knee方法并且在选定了的基础上 + 5个儿子，暂时不考虑非贪婪方法，逐层计算

- 逐层累计 其实每一层所选都差不多 因为每一层递进之后本身类内的数个变体会被拉开，K更像“类内变体数 / 局部结构数

- 脚本1到8层autoK表明autoK本身不错，但层数多了会相互牵制，观察到move偏向部分层级。假设如果独立多层会牵制，那么共享同一个 AAT layer 反复迭代1到8次就是一个很干净的对照实验，但是实验表明同一个 transport field 反复作用，会让样本不断被推离原始结构，出现过度搬运 / 动力系统退化。

- 层数问题暴露出来了，但是经过airline实验autoK也不能简单的聚类 还要考虑类内的交缠