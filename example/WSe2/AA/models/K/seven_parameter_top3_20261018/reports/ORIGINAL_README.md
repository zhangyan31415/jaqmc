# WSe2 top3 / Gamma repair, 2026-10-18

The primary deliverable is `wse2_top3_gamma_complete.pdf`. It compares the frozen previous fit with repaired candidates, includes six bands in every full band panel, highlights the top three, and documents endpoint deletion and small-angle sensitivity. This is a repair candidate, not a completed all-angle material model.

The Hamiltonian is unchanged from `wse2_seven_joint_20261017/model.py`: pure quadratic diagonal kinetic energy, first/second intralayer harmonics, first/second local tunneling harmonics, and one real linear-momentum tunneling amplitude. The seven physical coefficients are common mass, Vcos, Vsin, w1, V2, w2 and lambda. One mass and one lambda apply at **every angle within each fitted model**. No beta or extra physical term is added.

For the primary and strict candidates, each of the five potential/tunneling coefficients is A+B r^2, r=sin(theta/2)/sin(1.5 degrees). There are 12 global fitted coefficients. The quadratic diagnostic has 17 global coefficients but the same seven Hamiltonian terms. Exact A/B/C coefficients and units are in `angle_function_coefficients.csv`; portable code/configurations are under `models/ROLE/FOLD/`.

## What changed

Only the top three bands enter the spectral targets; only gaps12/23 enter the soft-loss diagnostics. The gap34 energy target, fourth-band projector and positive-gap34 acceptance veto are removed. There is no fixed-rank wavefunction veto, Chern target or inversion-sector constraint. The top3 projector is a weak tie-breaker, invariant under rotations inside that three-band subspace.

Every dataset has one Gamma-top1 energy alignment. No per-band or per-valley shift is fitted. The old angle-dependent bandwidth normalization and low-angle top1 enhancement are replaced with an absolute common 2meV energy scale and equal weights for all top3 bands.

The retained-angle minimax objective introduces an epigraph t >= mean_k,n<=3[(delta E_n)/2meV]^2 for each angle and minimizes t, with weak mean-error, top3-projector and angular-curvature tie-breakers. The following are separate constraints at **each retained angle**:

- Primary (`robust`): Gamma top2/top3 energy errors and Gamma gap23 error <=2meV; training Gamma-neighborhood top23 RMS <=2.5meV.
- Strict (`strict`): the same point/gap constraints <=1meV; training Gamma-neighborhood RMS <=2meV.

Optimization uses the original 36 training k points including Gamma/M/K. Spectral and Gamma-neighborhood metrics in the report use all saved path points, including historically inspected unused k points. The strict exploratory target of <=2meV full-path top3 RMS at every angle is still not fully met; optimizer convergence is not equivalent to passing all accuracy goals.

## Scope and selection

Own saved Heff data cover 13 commensurate angles, 2.13 to 9.43 degrees. The primary and strict candidate fit window is 2.13 to 5.09 degrees (10 angles). The 6.01, 7.34 and 9.43 degree panels are explicitly marked extrapolation. Separate all-angle and angle-independent-potential fits, including a widened V2 range, were attempted and still did not produce a reliable large-angle top3 spectrum. These failures are retained and not rebranded as a good transferable model.

The linear r-squared candidate is retained for simplicity and more stable endpoint-Gamma prediction. The quadratic r-squared candidate improves the mean interpolation error but gives much worse Gamma predictions after deleting 5.09 degrees. The Gamma-1meV candidate interpolates Gamma more closely but is more sensitive to deleting the high endpoint. Both alternatives remain visible rather than being silently discarded.

A weak projected top3-Heff-block residual was also tried. It uses only a 3x3 operator block and is invariant under consistent permutations/rotations of its reference eigenbasis; it does not add a Hamiltonian parameter. Its full-band endpoint prediction worsened, so it is not the primary model. Its attempted fits remain under `operator/`.

## Validation and limits

Each deletion refit removes the endpoint's electronic target data before initialization and optimization. Seeds use the same retained-angle soft fit, retained-only action data, or fixed generic constants. The excluded angle is evaluated only after fitting. Known cell geometry is an input, not a withheld electronic target. This is **retrospective** CV using a historically inspected dataset, not an unseen external benchmark.

Six-band paths are recomputed with enlarged complete radial Q shells until both valleys' Gamma-aligned path energies change by <0.01meV. The coarser fitting basis and converged spectra are compared explicitly. The reference is the stored K-valley Heff; opposite-valley curves follow time reversal, not an independently fitted Kprime reference.

The Gamma-only angular scan checks for a model Gamma-gap23 closing between sampled angles. It is not a full-BZ gap scan and cannot determine Chern numbers. The old model's Gamma closing near 4.524 degrees is a model prediction; the discrete reference dots alone do not establish the actual material's critical angle. No old model's Chern number is transferred to new fitted coefficients.

The bandwidth/gap figures use the saved Gamma-M-K-Gamma **path**, not full-BZ extrema. Below2.13 degrees no independent own-Heff truth is available. Small-angle curve spreads show model sensitivity, not uncertainty bounds or accuracy. Top4-6 plots are diagnostic context and were not fitted in this round.

## Reproduction

Environment `/data/home/zy/mambaforge/envs/moirekp`, Python3.11.15, authorized node `test001`. Each fit/evaluation command uses four worker processes and one BLAS/OpenMP thread per worker. Fit attempts, failed optimizer starts, inputs/configuration hashes, numerical checks and exact commands are recorded in the JSON files and `PROVENANCE.json`. No TAPW-to-Heff reconstruction/audit is performed in this study; the saved Heff inputs are treated as given.
