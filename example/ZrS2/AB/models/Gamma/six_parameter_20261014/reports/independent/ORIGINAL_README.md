# ZrS2 AB angle sweep, 2026-10-13

This experiment applies the existing Hanqi six-coefficient AB Hamiltonian family
to twelve available angles from 9.43 to 2.28 degrees. Both phases are fixed to
zero. It does not alter the original 3.15 fitted model.

`sweep.py` generates a 19Q native TAPW reference and fits Gamma-relative top-six
energies, three doublet projectors, native direct gap45 and development-sampled
indirect gap45, with weights all one. It uses three initial values (the final
3.15 model, the paper coefficients, and a 5% perturbation of the former), seed
20261012, analytic Jacobians and bounded SciPy TRF least squares.

The atomic common-anchor convention is transported from 3.15 to each angle.
Native local eigenspaces 40:48 are aligned by SVD; Schur energy is efermi minus
0.5 eV. 3.15 and 2.88 reuse their pre-symmetrized real-space input matrices.
Other angles use original SOC H/S with native TAPW C3 averaging. Each case
records input hashes, configuration and projection-to-native errors. This
transported convention does not constitute per-angle exact symmetry
certification. Any basis or projection failure is retained and reported.

TAPW native reciprocal labels are g and its plane waves use k+g; the formula
labels are Q=-g and use k-Q. The model also exchanges x/y coordinates.
The nonzero-M/K response regression test checks this sign against the frozen
3.15 convention. Initial wrong-sign candidates are retained solely in each
case's `invalid_qsign_diagnostic/`; their native reference matrices remain
valid and are reused. Final holdout coordinates were changed to untouched
points after this correction, and native overlap maps/Gram matrices are saved.

Path: 60 unique Gamma-M-K-Gamma points, plus thirteen development samples
around the native path's smallest direct gap. Development data select the
candidate; after freeze six new local and six new 2D native points are generated.
The quality gate is overlap >=0.93, local development gap RMS <=1.5 meV, and
sampled indirect gap >=1.5 meV. If no candidate passes, the best converged
validation-energy candidate is saved with `quality_gate=false`.

Execution uses `/data/home/zy/mambaforge/envs/moirekp`, Python3.11.15,
one process and eight BLAS threads per node. Direct SSH is user authorized.
The SSH sessions stay attached while the work runs; no detached children are used.

| Node | Cases, in sequence | Log |
|---|---|---|
| bigmem003 | 14_2.28, 3_9.43 | bigmem003.log |
| gpuh2002 | 13_2.45, 4_7.34 | gpuh2002.log |
| gpuh201 | 12_2.65, 5_6.01 | gpuh201.log |
| gpuh202 | 11_2.88, 6_5.09 | gpuh202.log |
| gpuh203 | 10_3.15, 7_4.41 | gpuh203.log |
| gpuh204 | 9_3.48, 8_3.89 | gpuh204.log |

Each `results/CASE/` contains `tapw.yaml`, `run.json`, incremental reference
packets, `candidates.json`, frozen `selected/`, fresh validation, and final
`result.json`. A completed case is skipped on a later run. Failed cases can
resume their saved reference points; no running case may be launched twice.

The projected-resolvent method was independently compared to explicit Schur
elimination (`test_numerics.py`). At3.15 Gamma the native reconstruction's Heff
differs from the frozen reference by4.48e-12 eV per matrix element and9.06e-14 eV
in the spectrum (`results/10_3.15/control.json`). This is a reduced control,
not evidence that all angles fit accurately.

Run `report.py` after inspecting job states to aggregate the case table and
scientific figures. Full2D gap guarantees are not inferred from sampled points.

Final run completed all12 cases and all six SSH workers exited successfully.
`REPORT.md` and `summary.csv/json` contain all parameter values, path/fresh
errors, native and model gaps, and projection-error decomposition.
`bands_all_angles.png/pdf` compares native top-six bands with fitted bands;
`parameters_vs_angle.png/pdf` marks the poor7.34/9.43 fits with red crosses.
Each case also has `gmkg_top12.png/pdf`, `extended_b1_low_energy.png/pdf`, and
an editable standalone `selected/model.py` plus `selected/model.npz`.

`VERIFICATION.json` independently recomputes held-out native energy/gap and
subspace-overlap metrics from the saved raw overlap maps and Gram matrices,
checks12 model hashes and disjoint holdouts, and confirms the original3.15
model hash is unchanged. `WORKERS.json` records the six workers' successful
final states. Solver termination alone does not imply an accurate fit: the
9.43/7.34 fits have large energy and wavefunction errors; several small-angle
fits have low energy errors but poorer local doublet projectors.
