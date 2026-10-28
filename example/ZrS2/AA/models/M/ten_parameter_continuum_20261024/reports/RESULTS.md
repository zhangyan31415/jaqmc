# ZrS2 AA M1 conduction model

10 physical coefficients, 26 independent global coefficients. Two inverse masses are exact constants across all 12 angles. Optimizer: `ftol` termination condition is satisfied..

| Angle | E RMS (meV) | gap1 relative RMS | gap2 relative RMS | P1 min | P2 min | P3 min |
|---|---:|---:|---:|---:|---:|---:|
|2.28|0.3995|0.0128|0.0518|0.9404|0.9174|0.8406|
|2.45|0.1778|0.0084|0.0247|0.9683|0.9488|0.9181|
|2.65|0.3217|0.0027|0.0284|0.9610|0.9568|0.9183|
|2.88|0.4352|0.0171|0.0469|0.9688|0.9667|0.9335|
|3.15|0.5588|0.0091|0.0587|0.9697|0.9733|0.9252|
|3.48|0.8991|0.0177|0.0472|0.9768|0.9726|0.9233|
|3.89|0.9844|0.0149|0.0732|0.9706|0.9696|0.8377|
|4.41|0.7399|0.0287|0.0400|0.9707|0.9451|0.8589|
|5.09|2.4210|0.0453|0.1089|0.9838|0.8454|0.6118|
|6.01|1.7862|0.0149|0.0619|0.9855|0.9180|0.7912|
|7.34|2.5181|0.0226|0.0755|0.9760|0.4538|0.4717|
|9.43|6.3140|0.0501|0.2094|0.9728|0.7157|0.6428|

All target energies and waves are reconstructed from the database viewer C3-symmetrized M1 TAPW H/S using its exact stored Q coordinates. Unsymmetrized cached waves are not used as reference. Native eigenvectors are source-only Rayleigh–Ritz orthogonalized before overlap evaluation. Source-derived fixed atomic/spin/origin conventions and deterministic per-k source downfolding are frozen before fitting. No C normalization or model-dependent alignment is applied.

Reference energy is a common first-pair Gamma center for each angle. The loss uses the native 3-pair energy window and native RMS pair-gap scales; the bandwidth of a single flat band is never used as a denominator.

Source summary and all fitting metrics: /data/work/zy/software/1.tapw_code/moirekp-release/examples/nonlinear_fit/material_nextbatch_20261024/ZrS2/fit_warped_10p_continuum10/RESULT.json

Expanded-Q checks test model sensitivity only; the native source is at the database cutoff. Their values are recorded in Q_checks; no source cutoff convergence claim is made.