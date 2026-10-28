# HfS2 AA M1 conduction model

9 physical coefficients, 23 independent global coefficients. Two inverse masses are exact constants across all 12 angles. Optimizer: `ftol` termination condition is satisfied..

| Angle | E RMS (meV) | gap1 relative RMS | gap2 relative RMS | P1 min | P2 min | P3 min |
|---|---:|---:|---:|---:|---:|---:|
|2.28|1.2278|0.0533|0.0725|0.9265|0.9349|0.7838|
|2.45|0.5814|0.0225|0.0608|0.8943|0.9349|0.8165|
|2.65|0.5015|0.0151|0.0320|0.9242|0.9496|0.8635|
|2.88|0.4795|0.0181|0.0271|0.9301|0.9604|0.8387|
|3.15|0.7193|0.0107|0.0425|0.9430|0.9599|0.8920|
|3.48|1.1978|0.0291|0.0405|0.9479|0.9610|0.7967|
|3.89|1.5901|0.0053|0.0844|0.9711|0.9505|0.5108|
|4.41|1.2617|0.0203|0.0641|0.9723|0.9347|0.7054|
|5.09|3.1627|0.0315|0.0951|0.9601|0.7645|0.4785|
|6.01|3.4505|0.0235|0.0810|0.9667|0.7820|0.6940|
|7.34|4.8681|0.0319|0.1082|0.9572|0.4193|0.2705|
|9.43|8.7226|0.0442|0.1957|0.9659|0.4383|0.4421|

All target energies and waves are reconstructed from the database viewer C3-symmetrized M1 TAPW H/S using its exact stored Q coordinates. Unsymmetrized cached waves are not used as reference. Native eigenvectors are source-only Rayleigh–Ritz orthogonalized before overlap evaluation. Source-derived fixed atomic/spin/origin conventions and deterministic per-k source downfolding are frozen before fitting. No C normalization or model-dependent alignment is applied.

Reference energy is a common first-pair Gamma center for each angle. The loss uses the native 3-pair energy window and native RMS pair-gap scales; the bandwidth of a single flat band is never used as a denominator.

Source summary and all fitting metrics: /data/work/zy/software/1.tapw_code/moirekp-release/examples/nonlinear_fit/material_nextbatch_20261024/HfS2/fit_warped_continuum9/RESULT.json

Expanded-Q checks test model sensitivity only; the native source is at the database cutoff. Their values are recorded in Q_checks; no source cutoff convergence claim is made.