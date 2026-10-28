# HfS2 AA M1 conduction model

10 physical coefficients, 26 independent global coefficients. Two inverse masses are exact constants across all 12 angles. Optimizer: `ftol` termination condition is satisfied..

| Angle | E RMS (meV) | gap1 relative RMS | gap2 relative RMS | P1 min | P2 min | P3 min |
|---|---:|---:|---:|---:|---:|---:|
|2.28|1.0153|0.0434|0.0654|0.9198|0.9339|0.7830|
|2.45|0.5127|0.0178|0.0515|0.8872|0.9339|0.8174|
|2.65|0.4725|0.0119|0.0201|0.9188|0.9500|0.8713|
|2.88|0.4289|0.0171|0.0264|0.9254|0.9600|0.8499|
|3.15|0.8500|0.0101|0.0512|0.9391|0.9614|0.9140|
|3.48|1.1895|0.0263|0.0393|0.9440|0.9635|0.8324|
|3.89|1.5947|0.0049|0.0844|0.9684|0.9520|0.4657|
|4.41|1.3207|0.0131|0.0617|0.9707|0.9376|0.7532|
|5.09|3.0343|0.0291|0.0993|0.9586|0.7603|0.4411|
|6.01|3.0948|0.0289|0.0869|0.9671|0.7976|0.6768|
|7.34|4.5645|0.0471|0.0770|0.9595|0.4099|0.3359|
|9.43|8.4890|0.0660|0.1780|0.9653|0.4375|0.1794|

All target energies and waves are reconstructed from the database viewer C3-symmetrized M1 TAPW H/S using its exact stored Q coordinates. Unsymmetrized cached waves are not used as reference. Native eigenvectors are source-only Rayleigh–Ritz orthogonalized before overlap evaluation. Source-derived fixed atomic/spin/origin conventions and deterministic per-k source downfolding are frozen before fitting. No C normalization or model-dependent alignment is applied.

Reference energy is a common first-pair Gamma center for each angle. The loss uses the native 3-pair energy window and native RMS pair-gap scales; the bandwidth of a single flat band is never used as a denominator.

Source summary and all fitting metrics: /data/work/zy/software/1.tapw_code/moirekp-release/examples/nonlinear_fit/material_nextbatch_20261024/HfS2/fit_warped_10p_continuum10/RESULT.json

Expanded-Q checks test model sensitivity only; the native source is at the database cutoff. Their values are recorded in Q_checks; no source cutoff convergence claim is made.