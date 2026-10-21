# ZrS2 AB 3.15：最终六参数模型的倒空间与实空间公式

本文对应 `projector_selected/model.npz`，参数向量为
`[-18.63731251867774, -8.810906733159781, 0.04747990715657983,
-0.04414463312395593, 0.018876345132669545, -0.020872143945069685]`，
依次是 `a1,a2,V1,V2,w1,w2`。前两项单位eV Å²，后四项单位eV。
模型形式沿用本地 arXiv:2605.13984 补充材料的AB简化式；本轮重新拟合
六个独立实系数，两个AB相位固定为零。

导出文件中的固定 `frame` 矩阵连接分子轨道基底与原投影基底。
它包含的基底相位不是下文的莫尔势相位phi1/phi2，也不是额外拟合变量。
以下实空间算符作用于莫尔包络场。

## 1. 基底、坐标与Q截断

每个Q处的内部基底为

$$
(tP\uparrow,tP\downarrow,t\bar s\uparrow,t\bar s\downarrow,
bP\uparrow,bP\downarrow,b\bar s\uparrow,b\bar s\downarrow).
$$

\(\tau_i,\sigma_i,s_i\)分别作用于层、分子轨道和模型自旋标签，
\(\Gamma_{ij}=\sigma_i\otimes s_j\)。\(\tau_z=+1/-1\)对应上/下层；
\(\sigma_z=+1/-1\)对应\(P/\bar s\)。模型自旋标签与微观物理自旋
算符的重建需使用已记录的投影/嵌入，不能仅由这些公式认定为未修饰的原子自旋。

公式坐标与前面作图所用的原始模型坐标相连：

$$
\binom{k_x}{k_y}=S\binom{K_X}{K_Y},\qquad
\binom{x}{y}=S\binom{X}{Y},\qquad
S=\begin{pmatrix}0&1\\1&0\end{pmatrix}.
$$

这是交换x/y的正交变换，行列式为−1。以下公式使用\((x,y),(k_x,k_y)\)。
第一谐波倒格矢取

$$
\mathbf g_1=g(1,0),\quad
\mathbf g_2=g(-\tfrac12,\tfrac{\sqrt3}{2}),\quad
\mathbf g_3=g(-\tfrac12,-\tfrac{\sqrt3}{2}),\quad
g=0.1086568701033\ \mathrm{\AA}^{-1}.
$$

可选原胞倒格基矢\(\mathbf b_1=\mathbf g_1\)、
\(\mathbf b_2=-\mathbf g_3=g(1/2,\sqrt3/2)\)。对应实空间基矢为
\(\mathbf a_1=(57.825936834,-33.385820197)\) Å、
\(\mathbf a_2=(0,66.771640395)\) Å。

$$
\mathcal Q_{19}=\{n_1\mathbf b_1+n_2\mathbf b_2:
\max(|n_1|,|n_2|,|n_1+n_2|)\le2\}.
$$

数值文件使用自身保存的Q排列。每个Q有8个内部态，总维数19×8=152。
之前长路径的\(\mathbf b_1^{\rm plot}\)变换后为\(-\mathbf g_3\)，
不应把它误认为这里沿x轴的\(\mathbf g_1\)。

## 2. 完整倒空间Hamiltonian

定义

$$
A_1=\Gamma_{yz},\qquad
A_2=\frac{\sqrt3}{2}\Gamma_{x0}-\frac12\Gamma_{yz},\qquad
A_3=-\frac{\sqrt3}{2}\Gamma_{x0}-\frac12\Gamma_{yz},
$$

$$
C_j(Q,Q')=\delta_{Q-Q',g_j}+\delta_{Q-Q',-g_j},\qquad
\mathbf p_Q=\mathbf k-\mathbf Q.
$$

每个\(Q,Q'\)矩阵块为8×8：

$$
\begin{aligned}
H_{QQ'}(\mathbf k)
={}&\delta_{QQ'}\Big\{
a_1p_Q^2\tau_0\otimes\Gamma_{00}
+a_2\tau_0\otimes[(p_{Qx}^2-p_{Qy}^2)\Gamma_{x0}
+2p_{Qx}p_{Qy}\Gamma_{yz}]\\
&\hspace{36mm}+V_1\tau_0\otimes\Gamma_{z0}
+w_1\tau_x\otimes\Gamma_{00}\Big\}\\
&+\sum_{j=1}^{3}C_j(Q,Q')
\left[V_2\tau_z\otimes A_j+w_2\tau_x\otimes\Gamma_{00}\right].
\end{aligned}
$$

所有未出现在这条公式中的矩阵元均为零。没有另外加入常数能量拟合项、
线性动量项、动量依赖第一谐波或自旋翻转项。

质量参数与动能系数的关系为

$$
a_i=\frac{\hbar^2}{2m_i^*}
=\frac{3.80998212}{m_i^*/m_e}\ \mathrm{eV\,\AA^2}.
$$

固定自旋\(s=\pm1\)后，单层Q对角块的2×2轨道矩阵为

$$
h_s(\mathbf p)=
\begin{pmatrix}
a_1p^2+V_1 & a_2(p_x-is p_y)^2\\
a_2(p_x+is p_y)^2 & a_1p^2-V_1
\end{pmatrix}.
$$

令\(A_j^{(s)}\)为上面\(A_j\)中的\(\Gamma_{x0}\to\sigma_x\)、
\(\Gamma_{yz}\to s\sigma_y\)，则每个自旋的完整层块为

$$
H^{(s)}_{QQ'}=
\begin{pmatrix}
\delta_{QQ'}h_s(\mathbf p_Q)+V_2\sum_jC_jA_j^{(s)} & T_{QQ'}I_2\\
T_{QQ'}I_2 & \delta_{QQ'}h_s(\mathbf p_Q)-V_2\sum_jC_jA_j^{(s)}
\end{pmatrix},\quad
T_{QQ'}=w_1\delta_{QQ'}+w_2\sum_jC_j.
$$

两个自旋块没有相互耦合；将基底重排为自旋优先时，完整Hamiltonian是
\(H^{(+)}\oplus H^{(-)}\)。

## 3. 傅里叶变换到实空间

采用与代码一致的平面波展开

$$
\Psi_{\mathbf k}(\mathbf r)=\frac1{\sqrt{A_M}}
\sum_{Q\in\mathcal Q_{19}}e^{i(\mathbf k-\mathbf Q)\cdot\mathbf r}c_{\mathbf kQ}.
$$

导数作用为

$$
-i\partial_a e^{i\mathbf p_Q\cdot\mathbf r}=p_{Qa}e^{i\mathbf p_Q\cdot\mathbf r},\quad
\partial_x^2-\partial_y^2\mapsto-(p_{Qx}^2-p_{Qy}^2),\quad
2\partial_x\partial_y\mapsto-2p_{Qx}p_{Qy}.
$$

周期势的矩阵元使用

$$
\frac1{A_M}\int_{\rm cell}d^2r\,
e^{i(Q-Q')\cdot r}\,2\cos(g_j\cdot r)=C_j(Q,Q').
$$

因此对应的实空间连续算符是

$$
\begin{aligned}
\mathcal H(\mathbf r)={}&
-a_1\nabla^2\tau_0\otimes\Gamma_{00}
-a_2\tau_0\otimes[(\partial_x^2-\partial_y^2)\Gamma_{x0}
+2\partial_x\partial_y\Gamma_{yz}]\\
&+V_1\tau_0\otimes\Gamma_{z0}
+2V_2\tau_z\otimes\sum_{j=1}^{3}A_j\cos(g_j\cdot r)\\
&+\left[w_1+2w_2\sum_{j=1}^{3}\cos(g_j\cdot r)\right]\tau_x\otimes\Gamma_{00}.
\end{aligned}
$$

这里导数作用于右侧波函数，势是乘法算符。也可用
\(\Psi_k=e^{ik\cdot r}u_k\)将其写为作用于周期函数的Bloch算符，
将\(-i\nabla\)替换为\(-i\nabla+k\)即可。实际152维模型是
\(P_{19}\mathcal H P_{19}\)的平面波矩阵。

为展开局部势，记\(c_j=\cos(g_j\cdot r)\)，

$$
F_x=\sqrt3(c_2-c_3),\qquad F_y=2c_1-c_2-c_3,\qquad
W=w_1+2w_2(c_1+c_2+c_3).
$$

固定自旋\(s\)后，在\((tP,t\bar s,bP,b\bar s)\)基底中有

$$
\mathcal H_s=
\begin{pmatrix}
D_+&B_{+,s}&W&0\\
B_{+,s}^{\dagger}&D_-&0&W\\
W&0&D_+&B_{-,s}\\
0&W&B_{-,s}^{\dagger}&D_-
\end{pmatrix},
$$

$$
D_\pm=-a_1\nabla^2\pm V_1,\qquad
B_{\eta,s}=-a_2(\partial_x-is\partial_y)^2+\eta V_2(F_x-isF_y),\quad\eta=\pm1.
$$

在周期边界下
\(B_{\eta,s}^{\dagger}=-a_2(\partial_x+is\partial_y)^2+\eta V_2(F_x+isF_y)\)。
这给出了全部实空间矩阵元。

## 4. 代入本轮参数

长度用Å、Hamiltonian用eV时：

$$
\begin{aligned}
\mathcal H={}&18.637312519\,\nabla^2\tau_0\otimes\Gamma_{00}\\
&+8.810906733\,\tau_0\otimes[(\partial_x^2-\partial_y^2)\Gamma_{x0}
+2\partial_x\partial_y\Gamma_{yz}]\\
&+0.047479907\,\tau_0\otimes\Gamma_{z0}
-0.088289266\,\tau_z\otimes\sum_j A_j c_j\\
&+[0.018876345-0.041744288(c_1+c_2+c_3)]\tau_x\otimes\Gamma_{00}.
\end{aligned}
$$

由于是价带模型，两个质量为负；正的\(\nabla^2\)系数作用在平面波上
仍给出负的动能主曲率。图中另外使用固定常数
\(E_{\Gamma,\max}=0.115004288934463\) eV将Γ带顶置零。
复现图时应减去该值乘单位阵，它不是第七个独立拟合参数。

## 5. 与Hanqi文章的比较

对照来源为本地
`paper/moirekp/arxiv_references/2605.13984/source/supplement_bare.tex`：
标签 `app:eqn:simplified model for AA/AB-ZrS2` 和
`app:tab:ZrS2_simplified_model_para_values`。

| 参数 | 文章AB，3.89° | 本轮AB，3.15 |
|---|---:|---:|
| m1*/me | -0.23760000 | -0.20442766 |
| m2*/me | -0.73870000 | -0.43241658 |
| V1 / meV | 43.230000 | 47.479907 |
| V2 / meV | -35.660000 | -44.144633 |
| w1 / meV | 12.080000 | 18.876345 |
| w2 / meV | -15.840000 | -20.872144 |
| phi1 / degree | 0 | 0（固定） |
| phi2 / degree | 0 | 0（固定） |

文章写成指数形式：

$$
V_2\tau_z\otimes\sum_{j,\alpha=\pm1}A_j
e^{i\alpha(\phi_1+g_j\cdot r)},\qquad
w_2\tau_x\otimes\Gamma_{00}\sum_{j,\alpha=\pm1}
e^{i\alpha(\phi_2+g_j\cdot r)}.
$$

AB中\(\phi_1=\phi_2=0\)，指数配对后正好等于本式中的2cos。
两个动能、V1、V2、w1、w2项逐项相同。
文章表列8项，AB只有6个自由实系数。本轮没有自动推导出另一套六参数
Hamiltonian：是在文章这套给定的简化形式内重新拟合。

相较文章，\(|a_2|\)增约70.83%，\(w_1\)增约56.26%，
\(|V_2|\)增约23.79%，\(|w_2|\)增约31.77%。这些差异同时包含几何、
参考数据与拟合目标的影响，不能直接解释为纯转角依赖。
\(m_1^*,m_2^*\)是两个动能结构的参数；仅对动能矩阵，其主曲率是
\(a_1\pm a_2\)，或\(1/m_\pm=1/m_1^*\pm1/m_2^*\)。

按层矩阵结构，V1是轨道在位分裂，V2是层对角莫尔势；w1/w2是层间耦合。
原TeX公式后的文字把V/w的层内外称谓写反，且w2求和下标存在排版问题；
这里按显式\(\tau\)矩阵和\(j=1,2,3\)求和解释，与已核对的运行代码一致。

## 6. 数值回算核验

在一个实空间原胞的16×16网格上，对上述8×8周期势做独立离散傅里叶积分，
再按导数对平面波的作用加入动能，重建152×152矩阵。Γ、M及一个普通k点
与冻结运行器的最大矩阵元差为约2.22e-16 eV。
核验数据在 `../../formula_derivation_20261013/verification.json`，
脚本为 `../../verify_formula_fourier.py`。没有修改模型系数或能带结果。
