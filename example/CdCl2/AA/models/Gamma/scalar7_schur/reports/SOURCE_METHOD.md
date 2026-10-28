# CdCl₂ 单CBM标量连续模型

原来的4通道heff保留了一对价带和一对导带，并非“两层导带轨道”。本模型只保留真正的CBM通道：参考矩阵C的前2Nq列。它是整个双层体系的一个离域CBM轨道，附带旁观者Kramers双重度；不能称作“每层一个轨道”。截取后不归一化C。

## 实空间和倒空间

\[
h_\theta=-A\nabla^2+\sum_{s=1}^{N_\star}\left[V_{sc}(\theta)c_s(\mathbf r)+V_{ss}(\theta)s_s(\mathbf r)\right],\qquad H=h_\theta\otimes I_2,
\]

\[
c_s=2\sum_{j=1}^3\cos(\mathbf G_{sj}\cdot\mathbf r),\qquad
s_s=2\sum_{j=1}^3\sin(\mathbf G_{sj}\cdot\mathbf r).
\]

用数据库实际倒格基矢B₀、B₁定义三组星：

- 第一星：B₀，B₁−B₀，−B₁；
- 第二星：B₀+B₁，B₁−2B₀，B₀−2B₁；
- 第三星：第一星的2倍。

它们的长度分别为G、√3G、2G。保留1、2、3个星，分别得到3、5、7个物理参数。A在全部12个角度中保持同一个拟合常数，单位eV Å²，有效质量m/m_e=3.80998211615486/A。

在exp[i(k+q)·r]约定中，动能为A|k+q|²δqq′；每个+G转移的势系数为V_sc−iV_ss，每个−G转移为V_sc+iV_ss。没有逐带能量偏移、SOC拟合项、层间隧穿或额外空间轨道。

## 固定坐标原点

最终版本使用`scalar*_schur`目录。每角度先对源Γ矩阵的自旋平均价带/导带分块，构造

\[
H^{\mathrm{origin}}_{c}=H_{cc}+H_{cv}(E_{\mathrm{CBM}}I-H_{vv})^{-1}H_{vc}.
\]

此矩阵只用于确定坐标原点，不直接作为连续模型哈密顿量，也没有把其中矩阵元当作隐藏物理参数。对其CBM第一星的三个复Fourier系数，用低q的固定Gaussian权重平均，再确定一次坐标原点：消除它们由平移造成的相位差，并在等价解中选择离原点最近者。这是源矩阵决定的坐标约定，不使用native/model波函数对齐，也不加入拟合参数。

若r₀是这个固定平移，则Dqq=exp(iq·r₀)，使用C′=C_CBM D和L′=L_CBM D进行模型比较。D是严格unitary；C未归一化。在原坐标中，模型波函数与势对应平移r→r+r₀。原始bare版本直接使用H_cc，其大角第一星相位包含被消去价带的影响，可产生不正确的源原点；Schur版本消除这一误差。各角度原点坐标保存在SETUP.json和RESULT.json中。

## 所有角度联合参数函数

设x=[sin²(θ/2)−sin²(θ_min/2)]/[sin²(θ_max/2)−sin²(θ_min/2)]，u=x/[0.2+0.8x]。每个势幅度用共同二次Bernstein函数

\[
V(\theta)=b_0(1-u)^2+2b_1u(1-u)+b_2u^2.
\]

A只有一个全局系数；其余物理参数各有三个角度系数。因此3/5/7参数模型分别拟合7/13/19个全局系数。坐标warp固定，不拟合。

## 训练范围预先由源容量确定

第一对带在全部12角度始终训练。第二、第三对只有在该角全部60个独立k点的CBM通道最小principal capture≥0.98时才训练。该判据在优化前固定。

因此7.34°、9.43°的第二、第三对不进入训练：它们基本不在这个CBM轨道子空间里。能带和overlap仍完整输出三对，这些未训练曲线用灰色空心点／阴影标记，不能把它们宣称为成功拟合。

## Loss

能量目标为TAPW原始自旋对中心，统一减去Γ最低对中心。每角度的三带整体窗口Sθ来自原始数据；gap尺度G_aθ为原始相邻对中心gap的路径RMS，数值下限0.01 meV。没有以单条平带带宽作分母。

每角度能量权重从[1,0.25,0.25]按有效带归一化；第一对优先。能量残差为δE/(0.01Sθ)。只有相邻两对都在训练范围时才加入相应gap残差δg/(0.05G_aθ)。角度和k点等权。

波函数残差为max(0,0.98P_ceiling−P)/0.04，并按同一带权重处理。P_ceiling来自固定源框架中spin-traced native density的最优标量表示上限；它只限定训练目标，不用于归一化图上的P。

训练projector使用固定0.01 meV展宽的固定秩Fermi占据，隐式微分化学势，以避免能级交叉处任意简并本征矢。最终图不用这个展宽：可分辨的native组显示直接P；模型边界穿过精确简并multiplet时，单对P设为不可唯一，并另用黑边方块显示完整model-cluster coverage。两种量不混入同一个raw-P最小值。

正则项为3||D(y−y_seed)||²+0.03||y−y_seed||²，其中D=[1,−2,1]、y为尺度化控制系数、seed来自原始native态作用量线性初始化。每个参数数目运行三个初值：source action、正弦反射以及已有bare模型延续（旧bare系列使用较小模型延续，3参数使用较深势初值）；选最好结果再收敛。初始与最终解析Jac均做差分核验。

## 图、数据及验证范围

每模型提供全部12角度、3对诊断能带、60个独立k点及闭合Γ重复、真实native overlap、全部参数曲线。扩Q检查在全部60独立k点重新对角化，并分别记录各带能量变化。它验证的是简化模型的截断敏感性，不等于重新完成原子TAPW的所有Q收敛。

本结果没有接入JAQMC或计算相互作用矩阵元。单CBM轨道的多体模型仍需明确相互作用投影及其适用带范围。

## 重现

环境：`/data/home/zy/mambaforge/envs/moirekp`，Python 3.11.15。计算节点gpuh201，每个拟合4个角度工作线程、每线程BLAS=1。源码基线commit为`e857d8bf9ba04e3dae8fdf093da7b34642e54469`；本批次脚本是工作区新增文件，具体源码与12份reference的SHA256记录在每个RESULT.json。该commit本身不能重建新增脚本，需保留脚本及哈希。

工作目录：`/data/work/zy/software/1.tapw_code/moirekp-release`。三个命令分别是环境Python运行`examples/nonlinear_fit/material_next5_models_20261023/cdcl2_scalar_fit.py --parameters N --origin schur`，N=3,5,7。每个输出目录有SETUP、MULTISTART、OPTIMIZATION、COMPLETE和RESULT记录；日志为CdCl2/scalarN_schur.log。PNG导出使用同环境的`cdcl2_scalar_export.py --origin schur`。
