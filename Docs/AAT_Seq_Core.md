AAT 序列结构的基本过程可以概括为：

$$
\boxed{
\text{状态}
\rightarrow
\text{AAT 评分}
\rightarrow
\text{评分表}
\rightarrow
\text{读取旧状态表}
\rightarrow
\text{写入新状态表}
\rightarrow
\text{形成搬运量}
\rightarrow
\text{合并}
\rightarrow
\text{残差搬运}
\rightarrow
\text{新状态}
}
$$

### 1. 基本符号

| 符号                  | 含义                          |
| ------------------- | --------------------------- |
| $t$                 | 序列中的当前位置                    |
| $D$                 | 模型的完整状态维度                   |
| $H$                 | head 数量                     |
| $d=D/H$             | 一个 head 的状态维度               |
| $x_t\in\mathbb R^d$ | 当前 head 在位置 $t$ 的状态         |
| $G$                 | 每个 head 内的 Ray Bank 数量      |
| $K$                 | 每个 bank 内的 ray 数量           |
| $R_k$               | 第 $k$ 根可训练 ray 的方向，归一化为单位向量 |
| $s_{t,k}$           | 位置 $t$ 对 ray $k$ 的原始 AAT 分数 |
| $a_{t,k}$           | 由读取算子产生的读取权重                |
| $w_{t,k}$           | 由对应记忆的写入算子产生的写入权重           |
| $M_{t,k}$           | 到位置 $t$ 为止，ray $k$ 中保存的状态   |

后续公式均描述一个确定的 layer、head 和 bank，因此省略 layer、head 和 bank 下标。

### 2. AAT 评分与评分表

对当前状态 $x_t$，先分离长度和方向：

$$
r_t=\lVert x_t\rVert,
\qquad
u_t=\frac{x_t}{\lVert x_t\rVert+\varepsilon}.
$$

将长度映射为归一化径向坐标：

$$
\rho_t
=

2\frac{r_t-r_{\min}}{r_{\max}-r_{\min}}-1.
$$

其中 $r_{\min}$ 和 $r_{\max}$ 是预先确定的尺度边界，而不是根据当前完整序列动态计算。

于是：

$$
x_t\rightarrow(r_t,u_t)\rightarrow(\rho_t,u_t).
$$

每根 ray $R_k$ 拥有：

* 方向 $R_k$；
* 径向敏感度 $q_k$；
* 偏置 $b_k$；
* 当前 head 共享的响应尺度 $\kappa$。

AAT 评分为：

$$
\boxed{
s_{t,k}
=======

\kappa e^{\rho_tq_k}
\left\langle u_t,R_k\right\rangle
+b_k
}
$$

所有 token 对所有 rays 的评分共同构成评分表：

| token $\backslash$ ray | ray 1     | ray 2     | ray 3     | $\cdots$ | ray $K$   |
| ---------------------- | --------- | --------- | --------- | -------- | --------- |
| $x_1$                  | $s_{1,1}$ | $s_{1,2}$ | $s_{1,3}$ | $\cdots$ | $s_{1,K}$ |
| $x_2$                  | $s_{2,1}$ | $s_{2,2}$ | $s_{2,3}$ | $\cdots$ | $s_{2,K}$ |
| $x_3$                  | $s_{3,1}$ | $s_{3,2}$ | $s_{3,3}$ | $\cdots$ | $s_{3,K}$ |
| $\vdots$               | $\vdots$  | $\vdots$  | $\vdots$  | $\ddots$ | $\vdots$  |

原始分数本身不是最终地址。读取算子和写入算子分别根据评分表产生：

$$
a_{t,k}=A(s),
\qquad
w_{t,k}=W(s).
$$

读取与写入不必使用相同的归一化方向：

* Content Memory 的写入沿 token 轴按列归一化；读取沿 ray 轴按行归一化
* Ordered Memory 使用当前 token 对 rays 的行地址，并按照自己的创新写入规则更新。

对于 Content Memory，其输入直接取当前状态：

$$
\boxed{q_t^{\mathrm{content}}=x_t}
$$

不再加入 $\bar x_{<t}$，也不在记忆内部额外设置 query 投影。它本身只表达内容关系，不额外编码顺序关系。

设 $\mathcal V_t$ 为当前位置允许看到的 token 集合

Content的写入和读出用的是：column Sparsemax write + row Softmax read。

在分类或双向任务中：

$$
\mathcal V_t={1,\ldots,T};
$$

在标准因果语言模型中：

$$
\mathcal V_t={1,\ldots,t}.
$$

因此，Content Memory 本身不负责表示顺序，但仍然服从具体任务规定的可见范围。

### 3. 状态表与状态读取

每个 bank 都维护独立的状态表：

| ray      | 当前保存的状态  |
| -------- | -------- |
| ray 1    | $M_1$    |
| ray 2    | $M_2$    |
| ray 3    | $M_3$    |
| $\vdots$ | $\vdots$ |
| ray $K$  | $M_K$    |

### 4. Head 与 Ray Bank

对于完整状态 $X_t\in\mathbb R^D$，首先通过一次完整状态投影产生 $H$ 个 head：

$$
\boxed{
[x_t^{(1)},x_t^{(2)},\ldots,x_t^{(H)}]
=

\operatorname{reshape}\left(W_{\mathrm{head}}X_t\right)
}
$$

其中：

$$
W_{\mathrm{head}}\in\mathbb R^{D\times D},
\qquad
x_t^{(h)}\in\mathbb R^d,
\qquad
d=\frac{D}{H}.
$$

每个 head 都由完整的 $D$ 维状态投影得到，而不是把原始状态直接硬切成若干部分。

例如：

$$
D=512,\qquad H=8,\qquad G=6,\qquad K=4.
$$

则：

* 一个 token 的完整状态为 512 维；
* 经过完整状态投影后得到 8 个 head；
* 每个 head 为 64 维；
* 每个 64 维 head 状态被直接复制到 6 个 bank；
* 每个 bank 仍然处理完整的 64 维状态，而不是把 64 维继续切成 6 份；
* 每个 bank 内有 4 根处于 64 维空间中的 rays。

不同 bank 使用独立初始化、独立训练的 rays 和状态表，因此即使输入状态相同，也会形成不同的评分、寻址和记忆结果。bank 的作用只是建立多个相互独立的竞争空间，第一版不在 bank 输入或输出处额外加入独立投影。

### 5. 整体结构

输入首先被映射到 $D$ 维状态空间，得到初始序列状态 $X_0$。当前 AAT 采用一条**持久记忆路径**和一条由多个同构 AAT Block 组成的**主状态路径**构成。持久记忆只在开始时建立一次：$X_0$ 首先经过一个独立的 Writer Block，其中依次执行 Content Memory 和 Ordered Memory

可以理解为它内部和主干 Block 一样正常分 head、分 bank、分 ray，因此最终对应的是很多独立状态表；后续每个主干 Block 读取持久记忆时，会同时使用整套这些 Content/Ordered 表，只是这些表建立一次后就固定不再更新。
$$
\widetilde X
=

\operatorname{WriterBlock}(X_0),
\qquad
\mathcal M^{\mathrm{persist}}
=

\operatorname{BuildMemory}(\widetilde X).
$$

它更像是在主干之外预先建立的一份长程状态表，之后保持不变，供所有 AAT Blocks 使用。

主状态路径由 $L$ 个参数相互独立、结构完全相同的 AAT Blocks 逐层堆叠。对于第 $\ell$ 个 Block，首先对当前状态 $X_\ell$ 做归一化和完整的 $D\rightarrow D$ 线性投影，再 reshape 为 $H$ 个 head。每个 head 的完整 $d=D/H$ 维状态同时进入 $G$ 个独立 Ray Banks；bank 不继续切分状态，也不额外进行 bank-specific 输入投影，而是通过各自独立的 rays 和状态表形成不同的评分与寻址空间。这样可以把大量 rays 分散到多个相互独立的竞争空间中，避免所有 rays 在同一个 Softmax 中进行过强的直接竞争。

每个 AAT Block 的计算顺序固定为：

$$
\large
X_\ell
\overset{\text{投影+分头/Bank}}{\longrightarrow}
\text{Persistent Read}
\overset{\text{合并+投影+残差}}{\longrightarrow}
X_\ell^{(1)}
$$
然后：
$$
\large
X_\ell^{(1)}
\overset{\text{投影+分头/Bank}}{\longrightarrow}
\text{Content}
\overset{\text{合并+投影+残差}}{\longrightarrow}
X_\ell^{(2)}
$$
最后：
$$
\large
X_\ell^{(2)}
\overset{\text{投影+分头/Bank}}{\longrightarrow}
\text{Ordered}
\overset{\text{合并+投影+残差}}{\longrightarrow}
X_{\ell+1}.
$$


首先，当前状态同时读取固定的 Persistent Content 和 Persistent Ordered 记忆。两种记忆产生的搬运量分别在每个 head 内跨 $G$ 个 banks 按 $1/\sqrt G$ 合并，再分别经过完整的输出投影；由于二者表达不同的记忆含义，投影后的结果最后以固定权重合并：

$$
\Delta X_\ell^{\mathrm{persist}}
=

\frac{
\Delta X_\ell^{\mathrm{content}}
+
\Delta X_\ell^{\mathrm{ordered}}
}{
\sqrt 2
}.
$$

完成持久记忆搬运以后，Block 执行自己的局部 AAT 计算。局部 Content Memory 先根据当前状态完成评分、读写和搬运；所有 banks 的结果在各自 head 内按 $1/\sqrt G$ 合并，heads 重新拼接并经过完整输出投影后，通过残差更新主状态。随后对**已经经过 Content 更新后的状态重新进行 head 投影**，再执行 Ordered Memory，并以相同方式完成 bank 合并、输出投影和第二次残差搬运。因此局部计算严格保持：

$$
X
\rightarrow
X^{\mathrm{content}}
\rightarrow
X^{\mathrm{ordered}}.
$$

最终可以将整个模型概括为：

$$
\boxed{
\begin{aligned}
\mathcal M^{\mathrm{persist}}
=
\operatorname{PersistentWriter}(X_0),
\quad
X_{\ell+1}
=
\operatorname{AATBlock}_{\ell}
\left(
X_\ell,
\mathcal M^{\mathrm{persist}}
\right),
\quad
\ell=0,\ldots,L-1.
\end{aligned}
}
$$

最后一层状态 $X_L$ 再交给具体任务对应的输出头。例如语言模型使用词表投影

### 因果序列约束

对于自回归序列，位置 (t) 只能访问位置 (t) 之前形成的状态表：
$$
M_{<t,k}=M_{t-1,k}.
$$
因此，每个位置严格按照“先读取旧状态，再写入当前状态”的顺序执行：
$$
y_t=\operatorname{Read}(x_t,M_{t-1}),\qquad
M_t=\operatorname{Write}(M_{t-1},x_t).
$$

### 记忆一：Content Memory —— “相似的内容写入相同的方向”

当前 token 状态 $x_t$ 直接用于 Content 评分：

$$
s^{\mathrm{content}}_{t,k}
=

\operatorname{AATScore}(x_t,R_k).
$$

**写入：**

对于每根 ray，沿评分表的 token 轴按列进行 Sparsemax，使竞争较弱的 token 得到精确的零权重：

$$
\boxed{
\left(
w^{\mathrm{content}}_{1,k},
\ldots,
w^{\mathrm{content}}_{T,k}
\right)
=

\operatorname{Sparsemax}
\left(
s^{\mathrm{content}}_{1,k},
\ldots,
s^{\mathrm{content}}_{T,k}
\right)
}
$$

它表示不同 token 对同一根 ray 的相对写入强度。

随后直接构造 ray $k$ 保存的状态：

$$
\boxed{
M^{\mathrm{content}}_k
=

\sum_{t=1}^{T}
w^{\mathrm{content}}_{t,k}x_t
}
$$

因此 Content Memory 不维护随时间递推的状态，而是根据当前评分表直接形成完整状态表。

**读取：**

对于当前 token，沿评分表的 ray 轴按行进行 Softmax 归一化：

$$
\boxed{
a^{\mathrm{content}}_{t,k}
=

\frac{
\exp\left(s^{\mathrm{content}}*{t,k}\right)
}{
\sum_j
\exp\left(s^{\mathrm{content}}*{t,j}\right)
}
}
$$

它表示当前 token 应该从不同 rays 中分别读取多少内容。

随后从 Content 状态表中读取：

$$
\boxed{
y_t^{\mathrm{content}}
=

\sum_k
a^{\mathrm{content}}_{t,k}
M^{\mathrm{content}}_k
}
$$

最后通过前面的通用搬运规则，使用读取结果 $y_t^{\mathrm{content}}$ 更新当前 token。

### 记忆二：Ordered Memory —— “取号机”

当前 token 先通过 Linear 产生：

- 用于寻址的 key：$k_t$；
- 真正要保存的 value：$v_t$。

key 经过 AAT 得到 Ordered 评分：

$$
s^{\mathrm{ordered}}_{t,k}
=
\operatorname{AATScore}(k_t,R_k).
$$

沿评分表的 ray 轴按行进行 Softmax 归一化：

$$
\boxed{
a^{\mathrm{ordered}}_{t,k}
=
\frac{
\exp\left(s^{\mathrm{ordered}}_{t,k}\right)
}{
\sum_j
\exp\left(s^{\mathrm{ordered}}_{t,j}\right)
}
}
$$

与此同时，从 key 生成一个逐维相位：

$$
\phi_t
=
\pi\tanh(k_t\odot c).
$$

其中，$c$ 是可训练的逐维尺度参数。把 value 绑定到这个相位上：

$$
x_t
=
\left[
v_t\cos\phi_t,
v_t\sin\phi_t
\right].
$$

可以把这理解为把同一个实值内容放进一个复数相位空间：

$$
v_t e^{i\phi_t}.
$$

这样，寻址权重相近但 key 不同的内容仍然可以通过相位进一步分离。

**读取：**

假设 token $t$ 到来以前，每根 ray 已经保存状态 $M^{\mathrm{ordered}}_{t-1,k}$。

使用当前地址从旧状态表中读出相位绑定后的内容：

$$
\boxed{
z_t
=
\sum_k
a^{\mathrm{ordered}}_{t,k}
M^{\mathrm{ordered}}_{t-1,k}
}
$$

把 $z_t$ 拆成实部和虚部：

$$
z_t
=
\left[
z_t^{\Re},
z_t^{\Im}
\right].
$$

再使用当前 key 的相位进行解绑定：

$$
\boxed{
y_t^{\mathrm{ordered}}
=
z_t^{\Re}\cos\phi_t
+
z_t^{\Im}\sin\phi_t
}
$$

因此，当前 token 读取的是自己到来以前已经形成的 Ordered 状态。

**写入：**

读取结果 $z_t$ 表示旧记忆在当前地址下已经能够表示的相位绑定内容。

当前内容中无法被旧记忆表示的创新为：

$$
\boxed{
\eta_t
=
x_t-z_t
}
$$

Ordered Memory 使用同一个地址进行写入，因此令：

$$
w^{\mathrm{ordered}}_{t,k}
=
a^{\mathrm{ordered}}_{t,k}.
$$

随后将创新沿当前地址补写回状态表：

$$
\boxed{
M^{\mathrm{ordered}}_{t,k}
=
M^{\mathrm{ordered}}_{t-1,k}
+
\frac{
w^{\mathrm{ordered}}_{t,k}
}{
\sum_j
\left(w^{\mathrm{ordered}}_{t,j}\right)^2
}
\eta_t
}
$$

归一化项抵消了使用同一地址写入和读取时产生的平方衰减。因此，使用相同地址重新读取更新后的状态表时，有：

$$
\boxed{
\sum_k
w^{\mathrm{ordered}}_{t,k}
M^{\mathrm{ordered}}_{t,k}
=
x_t
}
$$

最后通过前面的通用搬运规则，使用读取结果 $y_t^{\mathrm{ordered}}$ 更新当前 token。
