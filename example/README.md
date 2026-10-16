# 材料结构示例

按 `材料 / 堆垛 / 转角 / 结构状态 / 原文件名` 分类，供后续加入多个材料使用。
当前收录 MoTe₂ 同质双层 AA 结构：26 个转角，52 个 VASP 结构文件，约 12.93 MiB。
这是结构数据集；这里的文件尚未自动转换成 NQS 哈密顿量或训练配置。

```text
example/
├── README.md
├── index.csv                     # 逐文件索引：转角、原子数、来源、SHA256
├── manifest.json                 # 完整元数据及 OpenMX 几何核对记录
└── MoTe2/
    └── AA/
        ├── README.md             # 转角总表和 OpenMX 对应关系
        ├── 1.08455/
        │   └── rigid/
        │       └── 1.08455.vasp
        ├── 3.14966/
        │   ├── rigid/
        │   │   └── 3.14966.vasp
        │   └── relaxed_cell/
        │       └── 3.14966_relaxed.vasp
        └── ...
```

## 命名和来源

- 材料使用化学式目录，如 `MoTe2`、将来可增加 `ZrS2` 等。
- 堆垛独立分类，如 `AA`、`AB`。
- 转角保留来源文件名的五位小数，单位为度；这是来源标称值，未由弛豫后坐标重新测角。
- `rigid/`：未弛豫结构。
- `relaxed_cell/`：用户指定的 MLFF 晶胞弛豫结构，来源目录为 `2.relaxed_structures_cell_relaxed`。
- 所有结构文件原样复制，原文件名和内容保持一致；没有改晶胞、坐标、元素或原子顺序。
- 同一转角的不同来源版本保留原文件名，不自动覆盖或合并。6.00898° 的版本说明见材料 README。
- 原始绝对路径和内容校验值保存在 `index.csv`、`manifest.json`。

## 数据范围

rigid 来源：
`/data/work/zy/software/1.tapw_code/tapw/examples/1.triangular_lattice/1.homo/1.MoTe2/AA/2.twist_structure_MLFF/1.rigid_structures`

relaxed 来源：
`/data/work/zy/software/1.tapw_code/tapw/examples/1.triangular_lattice/1.homo/1.MoTe2/AA/2.twist_structure_MLFF/2.relaxed_structures_cell_relaxed`

OpenMX 对照来源：
`/data/work/zy/software/1.tapw_code/tapw/examples/1.triangular_lattice/1.homo/1.MoTe2/AA/9.openmx_qk__mlff__cell_relaxed`

共 27 个 rigid 文件（26 个转角），25 个 relaxed 文件（24 个转角）。
1.08455°、1.20286° 在指定 relaxed 来源中没有文件，仅提供 rigid，不补造结果。
OpenMX 目录对应 15 个转角、30 份 SOC/NSOC 输入，其晶胞和有序原子坐标均已逐项核对。

## 新增材料

继续使用同一层级即可，例如：`ZrS2/AB/3.15000/rigid/` 和 `ZrS2/AB/3.15000/relaxed_cell/`。
新增状态使用独立且含义明确的目录名，例如固定晶胞弛豫可用 `relaxed_fixed_cell/`。
实际新增时同时补充来源、原子数和 SHA256 索引，避免把不同弛豫方式或来源版本混为一类。
