# HfSe2 AA M1 model delivery

**Main candidate:9physical coefficients,23global fit coefficients,one effective conduction orbital per layer,two constant masses.** All12angles and60independent native k points per angle were fitted together. An additional10-parameter control is retained.

| Model | All-angle3-pair energy RMS (meV) | Gap1/2 relative RMS | <=3.89deg first-two energy RMS (meV) | <=3.89deg first-two group Pmin |
|---|---:|---:|---:|---:|
| 9p | 3.4648 | 2.98%/9.22% | 0.8140 | 0.9374 |
| 10p | 3.4147 | 3.57%/8.64% | 0.7887 | 0.9215 |

The9p model is retained as the simpler main candidate. The10th coefficient is a common-layer oblique cosine, not an extra orbital or derivative. It provides no uniform quality improvement.

The two fixed inverse-mass coefficients are 1.96345280 and 24.35081225 eV Angstrom²; the corresponding masses are 1.940450 and 0.156462 electron masses.

Physical-layer identity was checked with atomic-z projectors and the actual TAPW projector at2.28,3.89,9.43degrees. Source layer1 is lower, layer2 upper; bare channels are pure layer states. Resolvent dressing has at most1.35%opposite-layer weight in these audits. All720local frames select conduction indices[24,25], with24states below the source Fermi level, from Hf9.0/Se7.0 source data. No Cd or Gamma-valence reference enters this model.

The implicit identity is a source-anchored Kramers/pseudospin identity, not a demonstrated laboratory-spin Pauli identity. Raw SOC branches are plotted, while fitted targets are pair centers.

**Q limitation:** the fit uses exactly the database viewer Q at each angle. At2.28degrees, per-layer Q counts24→44→70 change the lowest-band shape by0.402→0.042meV and the second-pair relative energy by1.660→0.233meV atΓ/M/K. At5.09degrees, the second-pair change is5.530→0.052meV for14→30→52Q per layer. Thus the enlarged model spectrum trends toward stability but differs from the cutoff where parameters were fitted. Native source Q convergence and continuous-realspace NQS readiness are not established.

The first-two low-angle overlaps are useful; larger-angle higher-band errors remain visible and are not covered by a global PASS label. Figures use actual dense native degenerate-subspace overlap without interpolation or C renormalization.

Artifacts: figures/ALL_ANGLES_BANDS.png, figures/LOWEST_BAND.png, figures/OVERLAP.png, figures/PARAMETERS.png; the10p controls are under figures10/. Formula/loss:METHOD.md. Standalone NumPy real-space operator:continuum_model.py. Numerical data:MODEL.json and fit_warped/predictions/*.npz. Full provenance and process states:RUN_STATES.json and PROVENANCE.md.
**Continuous-space cutoff control completed:** `MODEL_CONTINUUM9.json` / `fit_warped_continuum9` jointly refits the same 9 physical coefficients on a two-star-expanded model basis. Its low-angle first-two RMS is0.8814meV and finite-Q-target group projection minimum0.9108. A further reciprocal star changes the three pair energies by at most0.014228meV across all720path points after a common Gamma alignment. The native TAPW source remains at the database cutoff; that source's larger-Q error is unknown. This is the more appropriate parameter set for testing the continuous real-space operator. Original finite-Q9/10models are preserved. See `CONTINUUM_CONTROL.md` and `figures_continuum9/`; standalone `continuum_model.py` accepts `MODEL_CONTINUUM9.json` unchanged.
