# 七项连续模型的项选择对照

所有模型每层一个轨道，H_orbital tensor I2。新的7参数沿用9参数的实际层动能轴，两质量本征值固定为全角度常数。保留两个隧穿与三个势项；未保留的两个势项严格为零。每角度7个物理参数，共17个独立角度系数。原7参数额外强制零扭转镜面对称并删掉两个层奇势；新7参数放松此近似，不再支持独立层宇称标量扇区。

先对10种三势项子集作固定系数零化筛选，再联合重拟合每材料前两名。筛选不是全局最优证明，选择也不是独立验证。全部12角度、60独立k沿用同一nativeTAPW能量/波函数目标、原始三带窗口及gap归一化、共同Γ零点、固定rho、正则化种子和权重。原始三带科学验收状态保留，低能范围指标单独列出。

|材料|模型|全角度三对中心RMS meV|≤6.01°前两对中心RMS meV|≤6.01°前两对最差P|≤3.89°前两对中心RMS meV|≤3.89°前两对最差P|
|---|---|---:|---:|---:|---:|---:|
|sns2_m|old7|4.851|2.962|0.3974|2.839|0.9176|
|sns2_m|constant_mass9|1.838|0.816|0.8669|0.573|0.9524|
|sns2_m|new7_rank0|3.826|1.753|0.8407|1.088|0.9228|
|sns2_m|new7_rank1|3.430|1.508|0.7933|0.827|0.8456|
|snse2_m|old7|6.268|3.873|0.6873|4.172|0.8573|
|snse2_m|constant_mass9|3.021|0.922|0.9170|0.727|0.9636|
|snse2_m|new7_rank0|2.924|1.330|0.9130|1.090|0.9234|
|snse2_m|new7_rank1|2.999|0.848|0.9146|0.606|0.9449|

sns2_m 当前展示的7项：A_parallel_eV_A2, A_perp_eV_A2, w_a_eV, w_b_eV, Sd_eV, Sa_eV, Sl_eV.
Prioritize converged candidates, then lowest-pair overlap floor .9 over theta<=6.02, then lowest-two-pair energyRMS over theta<=6.02; late third/fourth-pair error is reported separately. Training comparison only.

snse2_m 当前展示的7项：A_parallel_eV_A2, A_perp_eV_A2, w_a_eV, w_b_eV, Vd_eV, Sa_eV, Sl_eV.
Prioritize converged candidates, then lowest-pair overlap floor .9 over theta<=6.02, then lowest-two-pair energyRMS over theta<=6.02; late third/fourth-pair error is reported separately. Training comparison only.

9参数双质量固定版本一直保留：../m_constant_mass_20261021/{sns2_m,snse2_m}_9p/RESULT.json；仅固定第一项的版本以_parallel_only结尾。9个物理项的数量与质量是否随角度变化是两回事；双质量固定时有23个独立全局变量，物理项仍9个。

原始TAPW自旋分支与黑色pair center在现有纵轴下基本重合；2.28°最低一对单分支距中心最大SnS2=.00870014meV，SnSe2=.107729925meV。小于图上可辨宽度不等于严格自旋简并。完整逐角度自旋分裂见NATIVE_SPIN_SPLITTING.json。

全部静态图标注分带overlap。新的选中模型包含17个独立变量、源数据/代码SHA256、启动前源码快照、初始/终态残差梯度检验、扩展Q敏感性及可移植实空间势导出。导出对全部12角度×61点复现矩阵、能谱、group/principal overlap，另检查势周期性。还未重新收敛native-Q，也未接入JAQMC。

节点gpuh201-204运行四个联合拟合，gpuh2002导出；环境/data/home/zy/mambaforge/envs/moirekp，1进程×4角度线程，BLAS1。具体命令、日志和过程结束状态见各rank目录RUNNING/COMPLETE.json。
