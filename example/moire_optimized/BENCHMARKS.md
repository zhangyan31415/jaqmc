# Historical benchmark record

These measurements predate the portable packaging. They describe the recorded
research implementations and hardware, not a new benchmark of this commit.
All quoted steady timings use float32, batch 512, chunk 512, N18, one
determinant, the full four-layer 256/32 network, and KFAC inverse updates every
step. A full training step includes sampling, local energy, parameter gradient,
KFAC, and parameter update. Compilation is excluded from these step timings.

## MoTe₂: same chunk and same GPU

Each variant received 10 warmup updates and 200 synchronized, consecutive timed
updates in Slurm job **582629**, on **test001 / RTX 4090**. The “original” column
is the agreed separate kinetic/SOC, centered-gradient, LU-solver comparator;
it is not the earliest project's end-to-end wall-time baseline.

| Point | Original (ms) | Merged derivatives (ms) | + weighted gradient / Cholesky (ms) | + sparse derivatives (ms) | Original / sparse |
| --- | ---: | ---: | ---: | ---: | ---: |
| Filling 1, balanced | 131.55 | 100.55 | 78.07 | 60.51 | 2.17× |
| Filling 1, FM | 105.47 | 82.09 | 67.28 | 48.03 | 2.20× |
| Filling 2, balanced | 133.14 | 101.30 | 79.06 | 62.02 | 2.15× |

All comparisons retain chunk 512; chunk tuning is not counted as an algorithmic
gain. Full-batch complex-energy and update checks passed at their recorded
tolerances. Separate dense/sparse frozen checks in job **582645** retained all
sampled coordinates and angles, checked raw values before reductions, and
verified observables and density normalization. The sparse rule compatibility
adapter addresses the JAX 0.11 sparse-sum `out_sharding` interface.

These results do not establish a corresponding improvement in total time to a
converged state. Frozen evaluation, compilation, burn-in, checkpointing, and
host loop overhead have different timing scopes.

## Γ valley: three alternatives on an already merged baseline

Material: AB-ZrS₂, 3.15°, frozen six-parameter real-space Γ model, N18, model
spin counts `(9,9)`, four layer × orbital components per sector. The complete
`a1`/`a2` operator, mixed derivatives, and Ewald interaction are retained.

The baseline already shares kinetic derivatives and uses weighted gradients
and Cholesky KFAC. Runs used **gpu40903 / RTX 4090 index 0**, 1 GPU and 4 CPUs,
OMP 4 and BLAS/NumExpr 1. Each had 10 warmup and 200 synchronized training
updates. Paired results:

| Candidate | Paired baseline → candidate (ms) | Speed ratio | Job |
| --- | ---: | ---: | --- |
| Six-direction nested JVP | 201.70 → 158.68 | 1.27× | 582950 |
| Six-direction Forward Laplacian | 201.57 → 157.97 | 1.28× | 582963 |
| Six-direction sparse Forward Laplacian | 201.08 → 158.42 | 1.27× | 582969 |

The three choices are alternatives, so their ratios do not compound. Sparse
propagation did not add a measured benefit over the other two grouped methods.
First-run Laplacian and sparse-Laplacian means were 603.46 and 252.70 ms with
large fluctuations; one paired repeat per method gave the stable results above.
The original runs were retained. No root cause of that fluctuation was proven.
All eight run summaries, including those two runs, are retained in
[`validation/gamma_full_step_summary.csv`](validation/gamma_full_step_summary.csv).

The original first-step, full-512-sample FP32 comparisons recorded:

| Candidate | Maximum complex energy difference (meV/configuration) | Gradient relative L2 | Update relative L2 | Historical max-energy gate ≤ 0.01 meV |
| --- | ---: | ---: | ---: | --- |
| Nested JVP | 0.03559873 | 3.44e−6 | 6.37e−5 | Failed |
| Forward Laplacian | 0.01665097 | 1.73e−6 | 6.13e−5 | Failed |
| Sparse Forward Laplacian | 0.03576681 | 3.46e−6 | 6.30e−5 | Failed |

Mean-energy, gradient, and update checks passed; maximum single-configuration
energy checks did not. Initial parameters, positions, and angles were bitwise
equal. Smaller F32/F64 diagnostics passed separately and do not override the
512-sample gate. The user explicitly authorized packaging these candidates
without that historical gate blocking inclusion. The example defaults to JVP
for Γ, while the original JaQMC default path remains unchanged. The 1.27–1.28×
figures are measured execution-speed ratios, not demonstrated equal-accuracy
convergence speedups.

## Continuous angles versus discrete layers

The full MoTe₂ cold-start comparison tested filling 1, 3°, FM `(18,0)`, N18,
full network, batch/chunk 512, and the same random initial network and positions.
At the preserved 50,000-update budget both arms remained measurably improving;
neither met the declared late-energy plateau rule. The subsequent trajectory
was stopped by the user at 70,000 updates. No ground-state or complete phase
diagram claim follows from this study.

| Representation | Synchronized step (ms) | Full host-loop interval (ms) |
| --- | ---: | ---: |
| Continuous angles | 49.022 | 51.221 |
| Discrete layer labels | 51.158 | 54.053 |

These are equal-weight averages of forward-order and reverse-order 200-update
replays in job **583068**, on the same allocated GPU, starting from each arm's
45k state. The discrete version is approximately 4.4% slower in synchronized
compute and 5.5% slower in loop intervals. Replay data are outside the scientific
training trajectory. This is one point and one seed, not a universal ordering.

Independent common continuous-basis energy estimates at 50k were
−55.223574 ± 0.000969 (continuous-trained) and −55.221573 ± 0.001040
(discrete-trained) meV/hole, using conservative standard errors. Their endpoint
comparison passed the declared 0.01 meV/hole compatibility rule. That is
compatibility at the measured resolution, not proof of global optimization.
The original paper/configuration's batch 4096 differs from the tested 512.

## Provenance

The historical source baseline was commit
`266542b54b03733cfe72041795e2253bddcdf51a`. Compact numerical records and SHA256
hashes of the source summaries are in
[`validation/historical_benchmarks.json`](validation/historical_benchmarks.json).
Job IDs identify the original cluster records; they are not portable job
submission instructions. The original full checkpoints, HDF5 data, exact
commands, hardware logs, and failed-attempt records remain in the research
archive and are deliberately not copied into Git.
