# MoTe₂ CDW 首批复现计算

对象：Mishra、Wolf、MacDonald，arXiv:2603.05714v2。

本目录首先执行 **R1：完整双层连续模型的 NQS 检验**。论文相图使用
**R0：绝热映射、第一壳层、多 Landau level 自洽 HF**，两者不能直接比较
绝对能量或宣称定量复现。`adiabatic.py` 目前只生成 R0 场和 LLL 有效势，
尚无多 LL HF。FCI/CDW 相界与多体 Chern 数也尚未计算。

## 首批参数与初态

- 扭角：3.0°、4.2°；孔填充：1/3、2/3；27 原胞，9/18 孔，物理自旋全极化。
- 模型来源：Wang 等，arXiv:2304.11864v2，表 I 和 Eq. (1)；
  \(V=20.8\) meV、电子隧穿 \(w=-23.8\) meV、\(\psi=107.7^\circ\)、
  \(m^*=0.6m_e\)、单层晶格常数 0.352 nm。仅第一壳层，介电常数 10，
  不屏蔽的 2D Ewald 库仑及均匀中性背景。
- 明确采用 \(h_h(p)=-h_e(-p)^T\)，能量零点 \(E_\mathrm{ref}=0\)；
  原生 JAQMC 的孔隧穿系数因此为 **+23.8 meV**。
- 超胞矩阵：27 原胞 `[[6,3],[3,6]]`；12 原胞试跑 `[[4,2],[2,4]]`。
  倒空间 K 点与两种超胞均可公度。
- `first_points.json` 枚举 48 个独立训练：两角度 × 两填充 × 三初态 × 四种子。
  三初态命名 MX/XM/MM：1/3 表示占据位点，2/3 表示空缺位点；
  例如 2/3 的 MM 初态在 MX+XM 放置两孔。
  MX/XM 是所声明坐标、谷与层顺序下的约定，交换层会交换标签。
- 初态是周期 Gaussian Slater 的 Fourier 轨道系数，搭配小 FermiNet 残差。
  系数和神经网络均训练；Hamiltonian 无钉扎场。中心、Gaussian 宽度及
  初态自旋子不会在训练中成为约束。固定的短程 Jastrow 满足孔相互作用的
  接触斜率，后续需通过网络宽度、行列式数和 Jastrow 灵活性检查变分误差。
- 首批网络：32×3 单粒子隐藏层、16×3 双粒子隐藏层、1 行列式；
  supercell Fourier 指数截断 9，参数与坐标显式 float64。
- 优化：复波函数的实参数 SR，阻尼 0.05，量子度规/参数范数限制；
  每次测量之间 20 sweeps，连续层角按单粒子更新，空间按整构型更新。
  128 walkers，8000 步，每 2000 步独立冻结测量 512 次。
- 采样器默认在一次调用内缓存仅依赖位置的裸层轨道和 Jastrow；
  接受空间提议时同步更新缓存，角度提议继续使用原生多行列式 `LogDet`。
  每次调用重新计算缓存，SR 更新参数后不会沿用旧值。
  `model.make_sampler(cache_orbitals=False)` 可使用原始路径核对。
  已提交的作业使用提交时的源码归档，不会随 checkout 修改自动更新。

## 校验与运行

在 checkout 根目录设置 `PYTHONPATH=src`。Python 3.12，GPU 依赖见
`requirements.txt`；本地 CPU 环境使用 JAX/JAXLIB 0.9.1。

```bash
python -m pytest tests/app/moire/test_unprojected_charge_order.py \
  tests/app/moire/test_cdw_reference.py -q
python example/MoTe2/cdw_reproduction/run.py validate \
  --out /tmp/cdw-gpu-validation --walkers 16 --steps 5
python example/MoTe2/cdw_reproduction/run.py train \
  --out /tmp/cdw-first-points --rank 0
```

GPU 验证必须依次通过无相互作用 Slater 控制、正式网络规模的
12 原胞 4/8 孔 SR 试跑以及 scan/Forward Laplacian 能量一致性。
Slater 控制在有限平面波基组中精确对角化；局域能量仍有基组截断误差，
不能将有限截断的参考称为无限基组精确结果。
CPU 减小规模的验证仅为开发检查，不能替代 GPU 验证。

`prepare_job.py` 生成自含源码归档、SHA256 和可审核的 Merlin YAML；
生成 48 H20 训练配置必须提供已经通过的实际 GPU 验证 JSON。
`launch.py` 在六主机上各运行八个独立副本，将各 GPU 进程绑定至不同
CPU 核，每分钟备份 checkpoint、日志、模型和冻结测量至 HDFS。
没有 JAX 跨主机 collectives，副本间互不共享采样或 SR 数据。

## 输出与判据

每个副本保存 `model.npz`、源码 hash、`history.json`、
可续跑 `latest.pkl`、`frozen_*.npz` 和 `evaluations.json`。
冻结测量保留复局域能量、独立 walker 时间轴、169 个倒空间密度分量、
位置与层角，以便复查结构因子、密度、层极化和误差。

仅训练中用于 SR 的能量残差裁剪；报告及冻结测量均不裁剪。
误差检查使用 walker 均值、分块均值、自相关和能量/电荷 rank Rhat，
但仍需跨初态、跨种子平台检查和扩充变分空间。
`status.json` 中 `training_budget_finished` 明确 **不表示收敛**。
首批结果只能用于检查两侧是否出现对应 CDW 趋势；不能单独确定热力学
相边界、FCI、拓扑简并、Hall 响应或误差已达到论文水平。

将 HDFS 的各 `host*/` 输出下载到同一本地目录后，运行：

```bash
python example/MoTe2/cdw_reproduction/analyze.py \
  --root /path/to/collected-results --out /path/to/analysis
```

汇总产生 `summary.json` 和 `replicas.csv`，保留各 seed 的结果及缺失
rank，不自动合并分支或选择相标签。冻结测量先检查粒子数和有限性，
然后计算 \(S(q)=\langle|\rho_q|^2\rangle/N\)、连通结构因子、
\(|\langle\rho_q\rangle|^2/N^2\) 及各分量误差。初态名字不能直接视为
最终构型；连续层角的直方图也不能当作物理层占据。
时间统计排除跨冻结测量的间隔，初始编译不包含在稳态每步耗时中。

## 从首轮 checkpoint 重新验证收敛

`revalidate.py` 读取已完成的 8000 步 checkpoint，在新目录保存输出。
固定参数的 8000 步重测之后，继续原 SR 与全局学习率进度到 12000 步，
在 10000、12000 步再次冻结测量。每轮先热化 1000 次 sampler 调用，
再记录 4096 次；未通过诊断时沿同一冻结轨迹延长到最多 8192 次。
没有在延长测量期间更新波函数参数。

重新验证使用缓存采样器，并在空间更新中混合 60% 整构型 Gaussian、
30% 单粒子 Gaussian（步长乘 `sqrt(nelec)`）、10% 正负原胞平移。
三类提议在周期超胞上均对称，采用原波函数概率的 Metropolis 比率。
默认训练入口仍使用整构型更新；GPU 执行前逐项检查两类采样器的
缓存/原始路径轨迹，以及 checkpoint 上 scan/Forward Laplacian 能量一致性。

新的冻结文件只保存复能量分项、K 星与第一壳层 primitive 密度分量，
以及最终采样 bank，以控制长链存储量。粒子数 sum rule 逐轮检查。
统计采用独立 ArviZ 实现，保留能量 Rhat <1.01、电荷 Rhat <1.02，
增加 bulk/tail ESS、初始两组差异、前后半程差异及虚部均值检查。
误差取 ArviZ MCSE、原分块/自相关估计和独立 walker 均值误差的最大值。

8000→10000→12000 步的冻结均值用于单独评估固定网络容量下的平台。
该判断要求测量先通过混合检查、标准误不超过 0.1 meV/孔；
10000→12000 步能差门限为 0.1 meV/孔加两倍合成标准误。
即使通过此有限预算验证，也不自动认证容量、尺寸或多初态竞争已收敛。

`prepare_job.py revalidate --checkpoint-root /mnt/hdfs/cdw/.../trial_437196199`
生成六主机配置；仍需提供实际 GPU 验证 JSON。提交后由
`launch_revalidation.py` 在每 GPU 启动一个验证副本并备份新输出。

R0 数值校准仍有待解决的差异：当前场构造给出每原胞一量子通量，
LLL 第一谐波在 3° 与 4.2° 异号，但其零点约 3.48°（完整场）或
3.54°（先截断矢势），与论文约 3.7° 不一致。此结果是模型约定/截断
检查中的开放问题，不应视为 R0 复现成功；需核对作者的场截断和原始
参数后才能开展同 Hamiltonian 的定量 HF/NQS 比较。

## 来源

- https://arxiv.org/html/2603.05714v2 （正文及补充材料 S1–S27）
- https://arxiv.org/html/2304.11864v2 （连续模型、参数表、晶格常数）
- JAQMC 基底提交 `c5aff96`；Forward Laplacian 的 JAX 兼容修改参考
  `266542b` 分支中的 `3536f2e`，没有整体合并该并行开发分支。
