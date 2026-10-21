# SnSe₂ AA：新7参数 M 谷连续模型

本目录保存2026-10-21项选择比较中选中的 **snse2_m_rank1**，与当前展示的 NEW7 图一致。上传保留原始拟合、代码和图，不重新拟合系数。两项质量张量本征值跨全部12个角度保持常数，每层一个轨道，自旋为旁观简并。

## 哈密顿量与7个参数

电子导带模型，取最低本征值为低能带。长度Å、能量eV、Cartesian动量Å⁻¹。

$$
h=\begin{pmatrix}K_1+U_e+U_o&T\\T&K_2+U_e-U_o\end{pmatrix},\qquad H=h\otimes I_{\rm spin},
$$
$$
K_l=-A_\parallel(\mathbf e_l\cdot\nabla)^2-A_\perp(\mathbf e_l^\perp\cdot\nabla)^2,
\quad \mathbf e_l=\boldsymbol\kappa_l/|\boldsymbol\kappa_l|.
$$

两层分别使用源几何保存的实际旋转主轴。固定质量本征值不意味着两层的质量张量方向相同。未施加旧7参数的共同主轴／精确层镜面对称近似，也不能直接拆成独立层宇称标量扇区。

$$
U_e=2S_a\sin(\mathbf G_0\cdot\mathbf r),
\qquad U_o=2V_d[\cos(\mathbf G_a\cdot\mathbf r)-\cos(\mathbf G_b\cdot\mathbf r)]+2S_l[\sin(\mathbf G_a\cdot\mathbf r)+\sin(\mathbf G_b\cdot\mathbf r)].
$$
$$
T(\mathbf r)=2w_a\cos(\mathbf q_a\cdot\mathbf r)+2w_b\cos(\mathbf q_b\cdot\mathbf r),
\quad \mathbf q_a=(\mathbf B_1-\mathbf B_2)/2,\quad \mathbf q_b=(\mathbf B_1+\mathbf B_2)/2.
$$

参数顺序为 $(A_\parallel,A_\perp,w_a,w_b,V_d,S_a,S_l)$。前两项单位eV·Å²，后五项eV；另外两项严格固定为 $S_d=V_a=0$。这里的 S 表示正弦傅里叶势幅，不是自旋算符。无SOC或动量相关隧穿，最高空间导数为二阶。`physical.py` 是原始共享几何辅助文件，含其他未选用模型的函数；实际入口 `model.py` → `kernel.py` 只使用本节7项。

|常量|数值|
|---|---:|
|A_parallel / eV·Å²|5.221665191944|
|A_perp / eV·Å²|13.786341671707|
|m_parallel / m_e|0.729648872|
|m_perp / m_e|0.276359183|
|固定角度坐标 rho|0.18945221891436223|

质量由 m/m_e=3.80998211615486/A 换算。这是新7重新拟合的常量，不能替换成旧9参数中的数值。

其余五个耦合各有三个二次Bernstein角度控制系数，故独立全局系数为 **2+5×3=17**。完整角度函数、傅里叶符号、边界条件和倒空间表达式见 [公共公式说明](../../../../../M_VALLEY_SEVEN_PARAMETER.md)。`COEFFICIENTS.json` 保存全部控制系数，`CATALOG.json` 保存12个精确转角；标称范围2.28–9.43°。`geometry.npz` 与代码必须配套使用，不应仅凭拟合曲线外推未验证的几何。

## 使用：只需 Python 和 NumPy

在仓库根目录执行：

```bash
cd example/SnSe2/AA/models/M/seven_parameter_constant_mass_20261021
python - <<'PYTHON'
import numpy as np
import model
theta = model.record(3.89)["theta_deg"]       # 使用目录中的精确转角
print(model.parameters(theta))                # 该角度的7个系数，按上述顺序
print(model.full_parameters(theta))           # 展开到统一9项，未选项为0
print(np.linalg.eigvalsh(model.hamiltonian(theta, [0., 0.]))[:3])
print(model.realspace(theta, np.array([[0., 0.], [1., 2.]])))
print(model.kinetic_tensors(theta))           # 两层各一个2×2质量系数张量
print(model.enlarged_cell(theta))             # 使所有局部势周期的实空间胞，Å
PYTHON
```

`hamiltonian(theta,k_fractional)` 的k是源 `B` 的分数坐标，返回不显式复制自旋的轨道矩阵；`realspace(theta,r)` 的r为Cartesian Å，返回 `[U_layer1,U_layer2,T_real]`，单位eV。`parameters(theta)` 可计算连续角度函数；需要几何的接口仅接受12个目录角度附近的标签（源实现容差0.006°）。例如3.89对应3.8902381690076835°源几何；精确复现时使用CATALOG中的精确值。未实现空穴转换或新的原生JAQMC训练预设。

## 图、数据与适用范围

- [全12角度能带及P1/P2/P3](reports/ALL_BANDS.png)
- [全部7个参数曲线](reports/PARAMETERS.png) · [逐角度参数CSV](reports/parameters.csv)
- [选中候选及分带、分角度范围指标](reports/SELECTION.json)
- [原始项选择报告](reports/ORIGINAL_RESULTS.md) · [原始约定与拟合方法](reports/ORIGINAL_METHOD.md)
- [源导出核对](reports/SOURCE_EXPORT_VERIFICATION.json) · [源全部候选的验证记录](reports/ORIGINAL_VERIFICATION.json)

能带图黑线是native TAPW自旋对中心；彩色点为新7模型、颜色为overlap，橙虚线为旧共同主轴7参数，蓝点线为双质量固定9参数。此图中的细灰线连接新7模型中心；不是更早图中代表原始自旋分支的灰线。原始自旋分裂单独保存在 [NATIVE_SPIN_SPLITTING.json](reports/NATIVE_SPIN_SPLITTING.json)。图中Γ–M–K–Γ是莫尔小布里渊区路径；本模型描述底层 **M 谷**，不是Γ谷模型。

|范围|前两对带中心能量RMS / meV|P1 / P2 / P3 最低overlap|
|---|---:|---|
|≤3.89°标签（精确端点3.890238169°）|0.606460|0.9449 / 0.9482 / 0.7437|
|全12角度|2.299731|0.8682 / 0.7015 / 0.0033|

这些指标来自参与拟合和候选选择的已有数据，不是独立留出验证。较大角度第三／第四带局部错序作为范围边界记录；低能前两对带指标独立列出，不用一个全局最差overlap概括。原始全角度三带科学验收状态仍为 `FAIL`。模型Q扩展敏感性、native-Q收敛和小角度低能适用范围需分别评估；本次归档不重算Q收敛、不新增Chern或多体基态结论。`reports/predictions/` 保存12角度×61路径行的源能量、模型能量及分带overlap数据，末行Γ重复首行，独立k点为60。

## NQS 成本记录

[固定态成本测试](reports/nqs_cost/RUN.md) 在同卡A100、3.890238169°、18粒子、64 walkers、相同14064参数双层网络下比较了原生双层、旧7、新7和9参数。完整VMC/SR步新7约 **620.434 ms**，旧7约 **230.837 ms**；新7与9参数基本同价。主要增量是实际层质量轴对应的层相关二阶导数。该比值属于当前外部适配器实现的固定态计时，不是必然计算下界或收敛总工期。[逐轮记录](reports/nqs_cost/result.json) · [计时CSV](reports/nqs_cost/timings.csv)。原生JAQMC估计器未因本次归档自动支持该层相关质量张量。
