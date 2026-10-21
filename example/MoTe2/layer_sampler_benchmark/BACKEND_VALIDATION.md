# Same18-hole model: backend validation

Actual18-hole,27-cell,2-degree MoTe2 neural state:8854 real parameters, complex wavefunction,27 Gamma-projection translation copies. The input is a trained step2000 checkpoint, **not a certified ground-state checkpoint**. No toy model is used for the numbers below.

Hardware: one24GB RTX4090, one process, OMP/BLAS1. Python3.12.7, JAX/JAXLIB0.11.2, float64.128 saved walkers, energy/score microbatch8. Same parameter file, positions and angles for every fixed-state case.

Checkpoint SHA256: `81311dfc3791decfecbb0d83d400634394152e978f3ad6ea3b15aa8227f0655e`.
Bloch input SHA256: `3f0eb347a4e50e84d17ca91c97184c3c1a04ea4ff6c31e90a090747ed4f00fc6`.

## Numerical checks

| Check | Maximum absolute discrepancy |
|---|---:|
| All five energy components, meV/hole | 3.158e-14 |
| Spatial complex log-gradient | 4.670e-15 |
| Complex log-Laplacian | 6.355e-14 |
| Explicit-score vs VJP VMC force | 5.373e-14 |
| Force directional finite difference | 4.881e-11 |

KFAC directions are approximate and are not required to equal SR directions. The check compares their common force, includes phase and imaginary local energy, and verifies each timed update is a descent direction for that fixed clipped force. All15 parameter leaves are in10 repeated-dense blocks; generic blocks are rejected.

KFAC full-state pickle roundtrips preserve EMA, cached inverses and count. Separately, actual40-step training and20+20 resumed training match: positions/angles/RNG exactly, parameters to2.6e-15, optimizer state to4.4e-14.

## Steady-state update throughput

Every row includes local energy, force and optimizer update.30 synchronized calls after20 warmup calls; initialization/compilation excluded and retained separately in the raw JSON. KFAC inverse-period10 timing spans three inverse refreshes. Case order alternates.

| Configuration | Mean seconds/update | Median seconds/update |
|---|---:|---:|
| SR + scan | 0.694171 | 0.693728 |
| KFAC + scan, inverse every1 | 0.687747 | 0.687189 |
| KFAC + scan, inverse every10 | 0.685665 | 0.685690 |
| SR + forward Laplacian | 0.246005 | 0.245946 |
| KFAC + forward, inverse every1 | 0.239469 | 0.239530 |
| KFAC + forward, inverse every10 | 0.237322 | 0.237070 |

Pure local-energy median: scan 0.651893s; forward 0.217379s; 3.00x speedup.

## Actual short training loops

Both loops start from the same input checkpoint and sampler bank. Parameters, sampler and optimizer state then advance normally.30 steps, with compilation performed beforehand. These include orbital refresh,10 sampling sweeps, energy and update; the trajectories may diverge as expected.

| Configuration | Mean seconds/iteration |
|---|---:|
| SR + scan | 0.777486 |
| KFAC + forward, inverse every10 | 0.326974 |

Measured short-loop throughput ratio: 2.38x. This is **not** a comparison of time to a converged energy. Most of the throughput gain comes from forward Laplacian; KFAC alone provides only a few-percent improvement in this test. No confidence or convergence claim is inferred from that small difference.

## Regressions and scope

- 255 gather/shape/stack tests passed, including complex and sparse derivatives.
- Six example numerical tests passed: two SR/microbatch contracts and four KFAC force/curvature/tag/rejection contracts.
- Ten existing optimizer-only tag/curvature cases passed in float32/float64 activation modes. The complete molecular optimizer test module could not be collected in this minimal environment because its optional molecular workflow imports require upath/PySCF; it is not reported as a full-suite pass.
- The controller forwarding path and all four samplers with the new energy backend are checked separately before submission.

The example intentionally rejects unknown generic KFAC blocks. The native class keeps this opt-in for compatibility. The gather API fix uses the new extension location with an older-JAX fallback. Stack now uses exact linear derivative propagation; remaining scatter messages describe exact dense-Jacobian propagation, not a full-Hessian fallback.

Raw synchronized timings, compilation, numerical checks, dependency Git pin and source-file hashes: [backend_validation.json](validation/backend_validation.json). Actual training-resume checks: [training_resume.json](validation/training_resume.json). Reference weights are not included in the repository; the benchmark script accepts a matching trusted checkpoint and Bloch input from a run.

Re-run on the target GPU: these ratios are measurements on this4090, not predictions for other hardware.
