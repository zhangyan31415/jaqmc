# HfSe2 AA M1 conduction model

9 physical coefficients, 23 independent global coefficients. Two inverse masses are exact constants across all 12 angles. Optimizer: `ftol` termination condition is satisfied..

| Angle | E RMS (meV) | gap1 relative RMS | gap2 relative RMS | P1 min | P2 min | P3 min |
|---|---:|---:|---:|---:|---:|---:|
|2.28|0.4712|0.0233|0.0636|0.9374|0.9481|0.7036|
|2.45|0.2499|0.0092|0.0268|0.9532|0.9566|0.8583|
|2.65|0.4970|0.0164|0.0204|0.9639|0.9570|0.8459|
|2.88|0.6440|0.0140|0.0323|0.9672|0.9624|0.8440|
|3.15|0.9254|0.0053|0.0733|0.9782|0.9528|0.8531|
|3.48|1.1939|0.0159|0.0568|0.9813|0.9527|0.7490|
|3.89|2.2237|0.0180|0.0958|0.9847|0.9433|0.7310|
|4.41|2.3791|0.0167|0.1259|0.9894|0.9028|0.5694|
|5.09|3.0136|0.0285|0.0709|0.9904|0.8320|0.7296|
|6.01|4.4988|0.0299|0.1257|0.9808|0.6738|0.4404|
|7.34|5.1644|0.0578|0.1221|0.9767|0.4367|0.4815|
|9.43|8.6159|0.0603|0.1622|0.9194|0.4523|0.4348|

All target energies and waves are reconstructed from the database viewer C3-symmetrized M1 TAPW H/S using its exact stored Q coordinates. Unsymmetrized cached waves are not used as reference. Native eigenvectors are source-only Rayleigh–Ritz orthogonalized before overlap evaluation. Source-derived fixed atomic/spin/origin conventions and deterministic per-k source downfolding are frozen before fitting. No C normalization or model-dependent alignment is applied.

Reference energy is a common first-pair Gamma center for each angle. The loss uses the native 3-pair energy window and native RMS pair-gap scales; the bandwidth of a single flat band is never used as a denominator.

Source summary and all fitting metrics: /data/work/zy/software/1.tapw_code/moirekp-release/examples/nonlinear_fit/material_next5_models_20261023/HfSe2/fit_warped/RESULT.json

Expanded-Q checks test model sensitivity only; the native source is at the database cutoff. Their values are recorded in Q_checks; no source cutoff convergence claim is made.