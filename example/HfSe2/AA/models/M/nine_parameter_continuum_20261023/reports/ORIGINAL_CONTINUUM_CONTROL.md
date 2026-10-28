# HfSe2 continuous-space cutoff control

The same 9 local physical coefficients (23 global controls), two exactly constant masses, one conduction orbital per layer, and all 12 angles were refitted with two extra reciprocal stars. The original 9p/10p results are preserved.

| Quantity | Original finite-Q 9p | Expanded-model-Q 9p |
|---|---:|---:|
| All-angle 3-pair RMS (meV) | 3.4648 | 3.7400 |
| Relative gap RMS | 2.98% / 9.22% | 3.55% / 9.51% |
| <=3.89 degree first-two RMS (meV) | 0.8140 | 0.8814 |
| <=3.89 degree first-two group projection minimum | 0.9374 | 0.9108 |

Full-path further enlargement (third added star) changes the first three pair energies after common Gamma alignment by at most 0.014228 meV. This checks all 720 independent path points. It is a model-cutoff stability check, not native TAPW source-Q convergence.

The source C was injected at matching original Q coordinates, with zero extra columns and no renormalization. Color therefore means projection onto the original finite-Q native target. The larger-Q native source error is unknown.

Mass coefficients are 2.0005525474 and 24.6471069723 eV Angstrom squared, corresponding to 1.9044649045 and 0.1545813113 electron masses. All original energy, gap, and overlap terms remain in the joint fit. The optimizer terminated by ftol after 30 evaluations. Initial and final directional Jacobian checks passed; every saved prediction and projection was recomputed independently.

For a continuous-space NQS implementation this expanded-model-Q candidate is the appropriate cutoff control; original finite-Q fit parameters should not silently be reused as though their continuum spectrum were unchanged. The larger-angle higher-band accuracy limitations remain visible in the figures.

Figures: figures_continuum9/ALL_ANGLES_BANDS.png, LOWEST_BAND.png, OVERLAP.png, PARAMETERS.png. Data: fit_warped_continuum9/RESULT.json, VERIFICATION.json, predictions/. Final process state: PROCESS.json (gpuh201, exit 0; four angle threads, BLAS1).
