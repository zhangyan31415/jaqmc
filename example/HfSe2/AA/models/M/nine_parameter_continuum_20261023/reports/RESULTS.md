# HfSe2 AA M1 conduction model

9 physical coefficients, 23 independent global coefficients. Two inverse masses are exact constants across all 12 angles. Optimizer: `ftol` termination condition is satisfied..

| Angle | E RMS (meV) | gap1 relative RMS | gap2 relative RMS | P1 min | P2 min | P3 min |
|---|---:|---:|---:|---:|---:|---:|
|2.28|0.7085|0.0278|0.0880|0.9315|0.9108|0.6872|
|2.45|0.2987|0.0140|0.0414|0.9513|0.9376|0.8610|
|2.65|0.6793|0.0144|0.0226|0.9618|0.9497|0.8349|
|2.88|0.6930|0.0217|0.0406|0.9687|0.9593|0.8438|
|3.15|0.9491|0.0070|0.0774|0.9783|0.9507|0.8601|
|3.48|1.3184|0.0158|0.0511|0.9826|0.9556|0.7430|
|3.89|1.9902|0.0130|0.0908|0.9848|0.9391|0.7676|
|4.41|1.8635|0.0206|0.0816|0.9869|0.9270|0.5789|
|5.09|4.0499|0.0407|0.1084|0.9850|0.6482|0.5927|
|6.01|4.6522|0.0296|0.1134|0.9791|0.7220|0.4367|
|7.34|5.5709|0.0679|0.1322|0.9770|0.4238|0.4330|
|9.43|9.3363|0.0740|0.1768|0.9195|0.4686|0.4414|

All target energies and waves are reconstructed from the database viewer C3-symmetrized M1 TAPW H/S using its exact stored Q coordinates. Unsymmetrized cached waves are not used as reference. Native eigenvectors are source-only Rayleigh–Ritz orthogonalized before overlap evaluation. Source-derived fixed atomic/spin/origin conventions and deterministic per-k source downfolding are frozen before fitting. No C normalization or model-dependent alignment is applied.

Reference energy is a common first-pair Gamma center for each angle. The loss uses the native 3-pair energy window and native RMS pair-gap scales; the bandwidth of a single flat band is never used as a denominator.

Source summary and all fitting metrics: /data/work/zy/software/1.tapw_code/moirekp-release/examples/nonlinear_fit/material_next5_models_20261023/HfSe2/fit_warped_continuum9/RESULT.json

Expanded-Q checks test model sensitivity only; the native source is at the database cutoff. Their values are recorded in Q_checks; no source cutoff convergence claim is made.

Model basis expanded by two reciprocal stars. Native reference remains the original database finite-Q TAPW. C is embedded by exact original Q indices with zero added columns, without renormalization. Overlap is projection against the original finite-Q target, not validation of an expanded-Q native source.
