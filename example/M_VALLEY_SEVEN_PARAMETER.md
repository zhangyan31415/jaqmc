# SnS₂／SnSe₂ 新7参数 M 谷模型：公共公式与约定

入口：[SnS₂ AA](SnS2/AA/models/M/seven_parameter_constant_mass_20261021/README.md) · [SnSe₂ AA](SnSe2/AA/models/M/seven_parameter_constant_mass_20261021/README.md)。

这两个归档对应2026-10-21展示的选中候选，分别为 `sns2_m_rank0`、`snse2_m_rank1`。本次上传复制现有拟合，不重新优化，也不改变原生JAQMC求解器。每个模型描述一个选定的底层M谷，两层各一个轨道；其余两个M谷需通过相应C₃几何旋转生成，当前接口不同时构造三谷或谷间散射。

## 实空间电子哈密顿量

在源包络规范中，Cartesian位置单位Å，能量eV，取 $\mathbf p=-i\nabla$：

$$
h(\mathbf r)=
\begin{pmatrix}
K_1+U_e(\mathbf r)+U_o(\mathbf r)&T(\mathbf r)\\
T(\mathbf r)&K_2+U_e(\mathbf r)-U_o(\mathbf r)
\end{pmatrix},\qquad H=h\otimes I_{\mathrm{spin}}.
$$

$$
K_l=-A_\parallel(\mathbf e_l\cdot\nabla)^2
    -A_\perp(\mathbf e_l^\perp\cdot\nabla)^2,
\quad \mathbf e_l=\frac{\boldsymbol\kappa_l}{|\boldsymbol\kappa_l|},
\quad \mathbf e_l^\perp=(-e_{ly},e_{lx}).
$$

两项质量系数跨角度固定，层主轴仍使用每个源几何中的实际方向。质量张量

$$
C_l=A_\parallel\mathbf e_l\mathbf e_l^T
    +A_\perp\mathbf e_l^\perp(\mathbf e_l^\perp)^T
$$

在两层通常不同；不能因两项本征值相同，就将 $K_1,K_2$ 替换成同一个标量算符。换算为电子质量单位为 $m_i/m_e=3.80998211615486/A_i$，其中 $A_i$ 单位eV·Å²。

用统一9项记号表达势，未选项严格设为零：

$$
\begin{aligned}
U_e={}&2S_d[\sin(\mathbf G_a\cdot\mathbf r)-\sin(\mathbf G_b\cdot\mathbf r)]
      +2V_a\cos(\mathbf G_0\cdot\mathbf r)+2S_a\sin(\mathbf G_0\cdot\mathbf r),\\
U_o={}&2V_d[\cos(\mathbf G_a\cdot\mathbf r)-\cos(\mathbf G_b\cdot\mathbf r)]
      +2S_l[\sin(\mathbf G_a\cdot\mathbf r)+\sin(\mathbf G_b\cdot\mathbf r)],\\
T={}&2w_a\cos(\mathbf q_a\cdot\mathbf r)+2w_b\cos(\mathbf q_b\cdot\mathbf r),\\
\mathbf q_a={}&(\mathbf B_1-\mathbf B_2)/2,\qquad
\mathbf q_b=(\mathbf B_1+\mathbf B_2)/2.
\end{aligned}
$$

|材料|7个物理参数的实际顺序|严格为零的项|
|---|---|---|
|SnS₂|$A_\parallel,A_\perp,w_a,w_b,S_d,S_a,S_l$|$V_a,V_d$|
|SnSe₂|$A_\parallel,A_\perp,w_a,w_b,V_d,S_a,S_l$|$S_d,V_a$|

所有 $S,V,w$ 都是局部傅里叶势幅，单位eV。这里的 $S$ 不是自旋算符；没有新增轨道通道、SOC、动量相关隧穿或高于二阶的动能。自旋简并来自模型近似 $H=h\otimes I$，不意味着native SOC参考数据严格自旋简并。

三条层内倒格矢属于同一第一星，满足 $\mathbf G_a-\mathbf G_b=\mathbf G_0$。正负号和次序由随模型复制的 `physical.stars(B,kappa1-kappa2,kappa1+kappa2)` 固定。不能自行改成另一组星向量而保留原系数的符号。每个角度的实际 $B,Q,\kappa$ 都在对应 `geometry.npz` 中。

## 倒空间与接口

源平面波为 $e^{i(\mathbf k-\mathbf Q)\cdot\mathbf r}$。`hamiltonian(theta,k_fractional)` 使用行向量 $\mathbf k=\mathbf k_{\rm fractional}B$，返回层优先的无自旋轨道矩阵：先全部第一层Q，再全部第二层Q。其动能对角元为

$$
\delta_{QQ'}\left[A_\parallel((\mathbf k-\mathbf Q)\cdot\mathbf e_l)^2
+A_\perp((\mathbf k-\mathbf Q)\cdot\mathbf e_l^\perp)^2\right].
$$

定义 $U_l(\mathbf r)=\sum_Gu_l(G)e^{iG\cdot r}$，$\eta_1=+1,\eta_2=-1$，则正向量系数为

$$
\begin{aligned}
u_l(G_a)&=-iS_d+\eta_l(V_d-iS_l),\\
u_l(G_b)&=+iS_d+\eta_l(-V_d-iS_l),\\
u_l(G_0)&=V_a-iS_a,
\end{aligned}
$$

负向量取复共轭；层对角矩阵元为 $u_l(Q'-Q)$。层间矩阵元为

$$
h_{1Q,2Q'}=w_a\sum_{\sigma=\pm1}\delta_{Q-Q',\sigma q_a}
+w_b\sum_{\sigma=\pm1}\delta_{Q-Q',\sigma q_b},\qquad h_{21}=h_{12}^\dagger.
$$

`realspace(theta,r)` 的r是Cartesian Å，返回 `[U_layer1,U_layer2,T_real]`，单位eV；`kinetic_tensors(theta)` 返回两层的 $C_l$；`parameters(theta)` 返回各材料的7个系数；`full_parameters(theta)` 按统一顺序 `[A_parallel,A_perp,w_a,w_b,Sd,Va,Vd,Sa,Sl]` 展开，未选项为0。

## 周期胞与旧7参数的区别

隧穿含半倒格矢，全部局部矩阵元共同周期的胞由

$$
A_{\mathrm{enlarged}}=2\pi
\begin{pmatrix}q_a\\q_b\end{pmatrix}^{-T}
$$

给出，实空间格矢按行保存。此胞面积为 $2\,|\det(2\pi B^{-T})|$，由 `enlarged_cell(theta)` 返回。直接实空间求解时可用这个胞；若采用原始莫尔胞，必须保留相应层规范缝合／边界变换。不能在原始胞内任意强加所有分量普通周期边界。

旧7参数同时删除 $V_d,S_l$ 并将两层主轴取相同方向，才得到 $h_7=(K+U_e)I+T\tau_x$ 以及层宇称标量形式。当前新7保留层反对称势，并恢复实际层主轴；该化简不再成立。仅少两个拟合参数不意味着恢复了旧模型的求解简化。

## 角度函数与适用范围

令 $s=\sin^2(\theta/2)$，其中三角函数的角度由度数转换为弧度，

$$
x=\frac{s-s_{\min}}{s_{\max}-s_{\min}},\quad
u=\frac{x}{\rho+(1-\rho)x},\quad
p_j(\theta)=c_{j0}(1-u)^2+2c_{j1}u(1-u)+c_{j2}u^2.
$$

两个质量行分别满足 $c_{j0}=c_{j1}=c_{j2}$；其余5个耦合各3个控制点，共17个独立全局系数。rho为继承的固定角度坐标参数，不是第8个哈密顿量项；角度函数二次也不意味着四阶空间动能。

源目录覆盖12个角度，精确范围2.2810596097298803–9.430007907896421°。可移植代码只为这12个几何保存了Q和倒格矢；连续角度参数曲线不能单独认证任意中间角度或更小角度的能带精度。源Q数随角度和材料改变。

参考数据为已有native TAPW自旋对中心及固定基底中的源波函数信息。P1/P2/P3分别记录三对带的overlap；较大角度的第三／第四带错序是单独记录的适用边界，低角度前两对带指标另列。完整三带全角度验收状态仍为FAIL；候选选择与这些指标使用同一已有数据，不是独立验证。Q扩展敏感性也保留在COEFFICIENTS中，本次不声称重新完成native-Q收敛。

## NQS 计算成本的含义

2026-10-21的外部适配器固定态测试，在单A100、18电子、64 walkers、同一14064参数双层网络下，新7完整步约620ms；旧共同主轴7参数约231ms；完整9参数也约620ms。主要额外工作是层相关质量张量作用于多体波函数时的二阶导数。局部势部分约1.4ms。实际网络规模没有增加，这一约2.7倍开销属于当前实现，不是理论最低代价，也不是训练收敛工期。

各模型 `reports/nqs_cost/` 保存原始固定态记录和求值验证。它们记录独立组件实验，不会使原生JAQMC估计器自动支持本模型，也不代表新增了可直接运行的训练YAML。库仑介电常数10仅是统一成本测试条件，不是本次单粒子拟合的材料屏蔽参数。

## 验证归档

在仓库根目录、仅依赖NumPy运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python example/verify_m_seven_parameter.py
```

此脚本核对归档SHA256、12角度×61行能谱、角度参数、两层质量张量、实空间势周期与分带范围指标，并在一个角度通过独立傅里叶积分核对矩阵。它不需要源机器绝对路径，不运行拟合，也不重新生成native波函数；存档overlap的再现来自原始导出检查，本脚本只核对其归档内容和范围汇总。
