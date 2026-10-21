# 材料示例

每种材料使用相同的入口：`材料/README.md` → `堆垛/README.md` → `models/谷/模型版本/README.md`。
模型说明列出参数顺序与单位、候选版本、使用命令、能带与参数趋势，以及适用范围。

## 材料索引

| 材料 / 堆垛 | 谷和模型 | 已归档内容 |
|---|---|---|
| [MoTe₂ / AA](MoTe2/README.md) | [K，六参数联合拟合](MoTe2/AA/models/K/six_parameter_joint_20261016/README.md) | 9 个标定角度、角度函数、能带、参数和删角度对比；另有原子结构 |
| [WSe₂ / AA](WSe2/README.md) | [K，七参数 top3 模型](WSe2/AA/models/K/seven_parameter_top3_20261018/README.md) | 主候选及对照、端点删除、小角度敏感性、完整 PDF |
| [ZrS₂ / AB](ZrS2/README.md) | [Gamma，六参数模型](ZrS2/AB/models/Gamma/six_parameter_20261014/README.md) | 12 角度独立模型、小角度及全角度联合对照、3.15° gap 改进版、能带与参数趋势 |
| [SnS₂ / AA](SnS2/README.md) | [M，新7参数双质量固定](SnS2/AA/models/M/seven_parameter_constant_mass_20261021/README.md) | 选中rank0，实际层质量轴、12角度、分带overlap、完整公式和固定态NQS成本记录 |
| [SnSe₂ / AA](SnSe2/README.md) | [M，新7参数双质量固定](SnSe2/AA/models/M/seven_parameter_constant_mass_20261021/README.md) | 选中rank1，实际层质量轴、12角度、分带overlap、完整公式和固定态NQS成本记录 |

SnS₂／SnSe₂ 的 [公共哈密顿量、角度函数及边界约定](M_VALLEY_SEVEN_PARAMETER.md) 说明两个新7参数模型与旧共同主轴7参数的区别。两种材料保留的三项势不同；均为每层一个轨道、无显式SOC的M谷模型。

## 目录约定

```text
example/
├── README.md
├── MoTe2/
│   ├── README.md
│   └── AA/
│       ├── README.md
│       ├── <五位小数转角>/{rigid,relaxed_cell}/<原文件名>
│       └── models/K/six_parameter_joint_20261016/
├── WSe2/
│   ├── README.md
│   └── AA/{README.md,models/K/seven_parameter_top3_20261018/}
├── ZrS2/
│   ├── README.md
│   └── AB/{README.md,models/Gamma/six_parameter_20261014/}
├── SnS2/
│   ├── README.md
│   └── AA/{README.md,models/M/seven_parameter_constant_mass_20261021/}
├── SnSe2/
│   ├── README.md
│   └── AA/{README.md,models/M/seven_parameter_constant_mass_20261021/}
├── M_VALLEY_SEVEN_PARAMETER.md          # 新7参数的实空间/倒空间公式与适用范围
├── verify_m_seven_parameter.py         # 两种新7模型的可移植归档检查
├── continuum_models_manifest.json      # 模型来源、原路径、SHA256
├── continuum_models_verification.json  # 归档完整性和读取核对
├── index.csv                          # 原有 MoTe2 结构逐文件索引
├── manifest.json                      # 原有 MoTe2 结构元数据
└── verification.json                  # 原有 MoTe2 结构核对记录
```

- `models/谷/模型版本/` 包含独立代码、参数和 `reports/`；多个候选用 `candidates/` 分类，多角度模型用 `cases/原算例标签/` 保存。
- 模型转角保留来源标签和精度；不要把两位小数算例标签当作精确转角。跨角度函数是否存在，以各模型说明为准；ZrS₂ 的平滑联合拟合不等同于可外推的解析角度函数。
- `reports/` 保留源报告、PNG/PDF 和数值表。对照或失败候选有明确标识；不把优化收敛写成材料精度达标。
- 所有模型都属于单粒子模型归档，未自动转为 JAQMC 多体哈密顿量或训练配置。WTe₂、WS₂ 尚未纳入本次归档。

## 原子结构数据

目前仅 MoTe₂ AA 收录原子结构：26 个来源标称转角，52 个 VASP 文件，约 12.93 MiB。
按 `材料/堆垛/转角/结构状态/原文件名` 保存；`rigid/` 为未弛豫，`relaxed_cell/` 为指定 MLFF 晶胞弛豫。
原文件名、内容、晶胞、坐标、元素和原子顺序均保持不变；不同来源版本不自动合并。
1.08455°、1.20286° 缺少指定来源的 relaxed 文件，仅有 rigid。6.00898° 的两个 relaxed 版本及 SOC/NSOC 对应关系见 [MoTe₂ AA 说明](MoTe2/AA/README.md)。

rigid 来源：
`/data/work/zy/software/1.tapw_code/tapw/examples/1.triangular_lattice/1.homo/1.MoTe2/AA/2.twist_structure_MLFF/1.rigid_structures`

relaxed 来源：
`/data/work/zy/software/1.tapw_code/tapw/examples/1.triangular_lattice/1.homo/1.MoTe2/AA/2.twist_structure_MLFF/2.relaxed_structures_cell_relaxed`

OpenMX 对照来源：
`/data/work/zy/software/1.tapw_code/tapw/examples/1.triangular_lattice/1.homo/1.MoTe2/AA/9.openmx_qk__mlff__cell_relaxed`

旧的 `index.csv`、`manifest.json`、`verification.json` 只描述这批结构；模型新增不会改写其数量或核对结果。
原始绝对路径和 SHA256 分别记录在结构索引及模型清单中。
