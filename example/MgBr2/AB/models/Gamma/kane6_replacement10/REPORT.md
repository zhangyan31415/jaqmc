# MgBr2 六分量 Kane10：最终替换对照，未通过物理质量门槛

本轮仅检验 3.890238°，三有效轨道、六个 Kramers 分量、恰好十个实物理系数。完整公式、参数计数、源坐标及 NQS 限制见 [MODEL.md](../MODEL.md)。这不是全角度模型；没有参数随角度曲线，也没有 JAQMC 集成。

源空间本身的三对捕获率全路径最差约为 0.999881、0.999798、0.999726。源固定、原生 TAPW 态不旋转，C 不归一化。因此当前明显的波函数误差来自这个有限模型及其局部数值解，不能归因于旧源空间丢带或简单 overlap convention。

| 指标 | 最终值 |
|---|---:|
| 六条原生分支能量 RMS | 1.150713 meV |
| 三对中心能量 RMS | 0.972263 meV |
| 两个相邻中心 gap RMS | 0.920528 / 2.224389 meV |
| 三对 mean P 最差 | 0.582157, 0.273428, 0.121234 |
| 三对 principal P 最差 | 0.517175, 2.08824e-07, 1.15494e-10 |
| 正二阶主符号最小值 | 0.99064336 eV Å² |

前两对 mean P 已只有 0.582/0.273，足以判定当前质量不足，第三/第四对交叉不能为它们的误差开脱。principal P 单独报告，避免平均 overlap 掩盖一个 Kramers 分量几乎失败。

模型 k=51 的第 6/7 态间隙仅约 1.35×10⁻¹¹ meV。第三对 raw P 和 principal P 在该点均为 NaN 并在图中掩膜；unresolved=[[51,4,6]]。该处整个模型简并簇的覆盖率 0.288934 另存，绝不当作第三对 raw P。Γ/M/K 固定源 overlap 矩阵和有效轨道权重见 FIXED_FRAME_DIAGNOSTICS.json；它们仅用于解释，没有重新排列能带。

同一模型从100评估短试验继续 Gauss–Newton；其最佳坐标在755次调用处原样转交 L-BFGS-B。固定预条件仅由起点 Jacobian 列范数确定一次，不增加物理参数。L-BFGS-B 用310次函数调用、204次迭代，loss 从230.78170568降至210.35984991，按相对函数变化阈值停止。SciPy success=True 只表示该停止条件，不能称光滑驻点：缩放梯度∞范数0.01046，未缩放投影梯度0.82379，随机方向残差 Jacobian FD 相对差0.00425。

GRADIENT_AUDIT.json 显示，在2×10⁻⁶至2×10⁻⁹步长下，可行的解析投影“下降”方向正负两侧反而都升高目标；第六/第七态恰好交叉，硬排序的三对中心能量/自旋分裂损失形成不可微尖点。这是记录在案的相对函数停滞，既不是因100步预算用尽而宣告失败，也不是光滑全局最优证明。此对照没有达到可交付模型质量，不继续全角度推广。

参数 Δlower 停在设定下界1.01824 meV，新下/上 HH 正弦 SOC 势仅 0.006807 meV。真实 native 目标并未在这个局部解中保留 action 预筛所推荐的势强度；action 筛选也曾产生质量塌缩和伪低态，故 action 分数从未被当作本征态拟合成功。

Q=37→61 的三点、六分支最差中心化变化0.0088081 meV，61→91为0.0000439 meV。均匀 Hamiltonian 在覆盖全部可能负能区域的径向/角向扫描中最低值0、位置p=0；该测试和正主符号不能替代真实拟合质量。图、指标、参数、掩膜均来自最终同一组参数。

| 物理系数 | 最终值（单位见 MODEL.md） |
|---|---:|
| A_half_eV_A2 | 2.09020371 |
| A_HH_shared_eV_A2 | 6.56715457 |
| Delta_half_eV | 0.0532662113 |
| Delta_lower_HH_eV | 0.00101823764 |
| v_lower_eV_A | 0.520661988 |
| v_upper_eV_A | 0.274181976 |
| b_lower_eV_A2 | 0.650824453 |
| b_upper_eV_A2 | -2.38917102 |
| cos_diag_orbital2 | 0.0203769619 |
| sin_cross_12_imag_sigma_z | 6.80736004e-06 |

交付图为 BANDS_OVERLAP.png（灰色原生八分支、黑色三对均值、彩色模型六分支；下图实线 mean P、虚线 principal P）及 PARAMETERS_PILOT.png（单角度参数表）。图标题同时列中心与六分支 RMS，没有把中心均值当成自旋分支拟合。

复现：gpuh204，cwd=/data/work/zy/software/1.tapw_code/moirekp-release，环境/data/home/zy/mambaforge/envs/moirekp（Python3.11.15），4个k线程、每线程BLAS1。命令为该环境 python 运行 ../refine_lbfgs.py，日志 ../LBFGS.log。进程1004741退出码0且已确认不存在；期望产物已逐项检查。源提交e857d8bf9ba04e3dae8fdf093da7b34642e54469，候选文件未跟踪，精确脚本及源参考 SHA256 在 RUNNING.json。原 GN 的人为停止及数值继续关系保存在 ../replacement_refined/GN_INTERRUPTION.json。
