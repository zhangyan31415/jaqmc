# ZrS₂ AB: six-parameter Gamma-valley models

This archive preserves the existing 2026-10-12–14 fits, coefficients and figures. No refitting, new first-principles calculation or JAQMC many-body training was performed for the upload.

## Model and scope

The valley is **Gamma**, not M or K. Each Q has eight internal states: layer × orbital × spin. There are two physical layers and two orbital labels per layer. The basis order is Q → layer → orbital → spin. The strict AB Hamiltonian has exactly six adjustable real coefficients, ordered `a1,a2,V1,V2,w1,w2`; both AB phases are fixed to zero. The first two coefficients have units eV Å² and the other four eV. Cartesian input k is in Å⁻¹; the exported code applies its saved coordinate transformation. The 19-Q examples have dimension 152.

`a1,a2` are kinetic coefficients in the two-orbital Hamiltonian. They must not be substituted for the one-mass scalar kinetic term used in the MoTe₂ or WSe₂ examples. Full real/reciprocal-space expressions for the separate 3.15° gap candidate are in [FORMULAS.md](candidates/projector_gap_3p15/cases/10_3.15/FORMULAS.md); its numerical coefficients belong to that candidate only.

## Candidates

| Directory under `candidates/` | Angles | Role |
|---|---|---|
| `independent/` | 12 labels, 2.28–9.43° | Original independently selected angle-by-angle baseline; inspect each result's quality gate. Large-angle failures are retained. |
| `small_angle_joint/` | 2.28, 2.45, 2.65, 2.88, 3.15° | Smoothness-weighted joint-fit diagnostic, `lambda_10_quadratic`. It does not replace the independent baseline. |
| `all_angle_joint/` | All 12 labels | Joint-fit diagnostic, `lambda_100_previous`, shown in the all-angle comparison. It does not repair the 7.34/9.43° mismatch. |
| `projector_gap_3p15/` | 3.15° only | Separate projector/gap-preserving alternative, with a band-accuracy tradeoff. It is not the 3.15° member of either joint fit. |

Every candidate has `parameters.csv` and `cases/<original-case-label>/model.py` plus `model.npz`. Original two-decimal angle labels are preserved rather than converted to fabricated five-decimal precision. `independent` also includes original selection/result records, a small comparison packet and source-run provenance for every case. Large native H/S matrices and wavefunction reference packets are not included.

These ZrS₂ joint fits optimize six coefficients **at each included angle**, with a parameter-curvature penalty. They do not export one analytic angle function or a common mass. Plotting lines joining fitted angle points are not a validated extrapolation law. The plot labels lambda=10/100 are loss regularization weights, not new Hamiltonian terms.

## Use

Python, NumPy and SciPy are required. From the repository root:

```sh
cd example/ZrS2/AB/models/Gamma/six_parameter_20261014/candidates/independent/cases/10_3.15
python -c "from model import SimplifiedModel; import numpy as np; m=SimplifiedModel('.'); print(m.parameters); print(np.linalg.eigvalsh(m.hamiltonian([0.,0.]))[-6:][::-1])"
```

Change the candidate and case directory to load another archived model. `SimplifiedModel.hamiltonian(k)` returns the full spinful electron Hamiltonian. `bands(kpoints)` returns eigenvalues in ascending order. Supplied `model.npz` geometry and basis ordering must stay paired with its code. A source-relative `SOURCE_RUN.json` is provenance, not a portable fit configuration.

## Reports and figures

| Result | Bands | Parameter trends | Full report |
|---|---|---|---|
| Independent | [12-angle top6](reports/independent/bands_all_angles.pdf) | [Six coefficients](reports/independent/parameters_vs_angle.pdf) | [Report](reports/independent/REPORT.md) |
| Small-angle joint | [Five-angle top6](reports/small_angle_joint/joint_top6_bands.pdf) | [Independent vs joint](reports/small_angle_joint/joint_parameter_trends.pdf) | [Report](reports/small_angle_joint/REPORT.md) |
| All-angle joint | [12-angle top6](reports/all_angle_joint/all_angle_joint_top6_bands.pdf) | [Independent vs joint](reports/all_angle_joint/all_angle_joint_parameter_trends.pdf) | [Report](reports/all_angle_joint/REPORT.md) |
| 3.15° gap candidate | [Band comparison](reports/projector_gap_3p15/projector_bands_comparison.pdf) | Single-angle candidate | [Gap tradeoff](reports/projector_gap_3p15/PROJECTOR_REPORT.md) |

PNG versions are alongside the PDFs. [3.15° full-path bands](reports/projector_gap_3p15/projector_gmkg_full.pdf) and [gap comparison](reports/projector_gap_3p15/projector_gap_comparison.pdf) are also included. Report JSONs retain the original numerical evidence and limits.

## Validation and limits

The sweep reference is native TAPW, with transported fixed basis conventions; it is not a new DFT calculation at upload time. Independent fits are substantially worse at 7.34/9.43°. Smoothing improves development/path averages slightly but does not improve all untouched native-point metrics, so the joint fits remain diagnostics. Some smaller-angle doublet overlaps also fall below the original target; optimizer convergence alone is not an accuracy certificate.

The 3.15° gap alternative has path native RMS 1.56142 meV and minimum doublet overlap 0.93248 in its original report. Dense minimum direct gap 2.31053 meV and sampled indirect gap 2.30031 meV are sampled diagnostics, not a full-BZ proof. No new Chern numbers, small-angle extrapolation accuracy or JAQMC training preset is claimed. Source formulas/readmes are retained unchanged; their original relative paths describe the source layout. The links in this README describe this archive's actual layout.
