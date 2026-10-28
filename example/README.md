# 材料示例

每种材料使用相同的入口：`材料/README.md` → `堆垛/README.md` → `models/谷/模型版本/README.md`。
模型说明列出参数顺序与单位、候选版本、使用命令、能带与参数趋势，以及适用范围。

## 当前模型与 HF 扫描交接（2026-10-29）

目前收录 **16种材料、37个模型版本**，其中34个版本提供统一的单粒子矩阵接口。
最新WSe₂17角度模型、MoTe₂10角度模型，以及原先缺少的Γ／M谷材料均已补入。
每角度物理参数与角度函数的全局系数分开记录；失败候选、单角度试验和高参数谱模型均保留明确标记。

- [完整材料／版本索引](hf_scan/MODELS.md) · [机器可读JSON](hf_scan/MODEL_CATALOG.json) · [CSV](hf_scan/MODEL_CATALOG.csv)
- [HF交接：读取命令、单位、基底、边界及相互作用输入](hf_scan/README.md)
- [逐包文件SHA256](hf_scan/ARCHIVE_FILES_20261029.json) · [统一入口验证](hf_scan/VERIFICATION.json)

| 材料 | 堆垛／谷 | 主要模型及范围 |
|---|---|---|
| [CdCl₂](CdCl2/README.md) | AA／Γ导带 | 3／5／7参数，单空间轨道；第一带候选 |
| [ZrS₂](ZrS2/README.md) | AA／M导带；AB／Γ旧模型 | 新10参数小角前两对；旧AB六参数是不同目标 |
| [HfSe₂](HfSe2/README.md) | AA／M导带 | 9参数扩模型Q，小角前两对 |
| [HfS₂](HfS2/README.md) | AA／M导带 | 9／10参数扩模型Q，小角前两对 |
| [SnS₂](SnS2/README.md)、[SnSe₂](SnSe2/README.md) | AA／M导带 | 新7参数、双质量固定9参数及质量约束对照；模型Q限制需保留 |
| [InSe](InSe/README.md) | AB／Γ导带 | 9／10参数两层，小角候选及预算限制 |
| [GaSe](GaSe/README.md)（Ga₂Se₂） | AB／Γ导带 | 两层10参数、单成键轨道4／6参数；全角精度有限 |
| [MgCl₂](MgCl2/README.md)、[Ga₂S₂](Ga2S2/README.md) | AA／Γ及AB／Γ导带 | 9／10参数诊断；非局域动能、质量及波函数限制明确列出 |
| [PtSe₂](PtSe2/README.md) | AA／Γ价带区 | 两种10参数四分量模型，全部12角；质量未通过 |
| [MgBr₂](MgBr2/README.md) | AB／Γ | 六分量价带10参数单角试验；另列单轨道导带3参数目标 |
| [WSe₂](WSe2/README.md) | AA／K价带 | 7参数17角主模型；历史版与后续失败角度律另列 |
| [MoTe₂](MoTe2/README.md) | AA／K价带 | 6参数10角，多种角度约束及能量／波函数局部解对照 |
| [WS₂](WS2/README.md)、[2H-WTe₂](WTe2/README.md) | AA／K | 单角数百系数谱模型，诊断用途；不是≤10参数模型 |

统一入口只导出单粒子矩阵，**不实现HF求解、不自动选择相互作用、不把全部模型接入JAQMC多体训练**。
微观密度form factor、谷／自旋味数、填充、介电／栅极环境、k网格与截断收敛由HF实现另行明确。
科学上未通过的候选仍可复现，但不能因“已归档”而作为默认合格扫描输入。

## 先前归档入口（保留历史路径）

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
- 所有模型都属于单粒子模型归档，未自动转为 JAQMC 多体哈密顿量或训练配置。WTe₂、WS₂已在2026-10-29补入一般谱诊断模型，详见上方当前索引。

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
