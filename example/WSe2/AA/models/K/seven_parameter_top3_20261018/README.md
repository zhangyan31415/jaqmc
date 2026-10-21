# WSe₂ AA: seven-parameter K-valley top3 model

Portable models and final comparisons from the 2026-10-18 Gamma/top3 refinement. No refit or new DFT calculation was performed for this upload.

## Recommended entry point

Use `candidates/robust/full/angle_model.py` and its adjacent `model.py` and `FUNCTION.json`. This is the selected primary candidate: pure quadratic diagonal kinetics, two intralayer Fourier shells, two local tunneling shells, and one real linear-momentum interlayer tunneling coefficient. The seven physical coefficients are `a_h, Vcos, Vsin, w1, V2, w2, lambda_eV_A`. Mass and lambda are shared across all angles: 0.342128286177495 m_e and 0.057115663561416986 eV Å.

The five potential/tunneling coefficients are linear in `r²`, where `r=sin(theta/2)/sin(3°/2)`. There are 12 global fitted coefficients. No beta or higher-order kinetic term is included. Input energies are eV and Cartesian momenta Å⁻¹; the configuration contains the supplied geometry conventions.

From the repository root, with Python and NumPy:

```sh
python example/WSe2/AA/models/K/seven_parameter_top3_20261018/candidates/robust/full/angle_model.py --angle 3.89
```

From `candidates/robust/full/`:

```sh
python -c "from angle_model import AngleModel; import numpy as np; m=AngleModel(); print(np.linalg.eigvalsh(m.hamiltonian(3.89,[0.,0.],nq=37))[-3:][::-1])"
```

The `nq=37` command is a small-basis usage example, not the converged spectra shown in the report. `hamiltonian(theta,k,valley='K',nq=169)` also accepts `valley='Kprime'`, implemented as time reversal `H_K(-k)*`. Bands at the same k need not agree across valleys; the reference is the stored K-valley Heff, not independently fitted Kprime data. Complete radial shells may contain more vectors than requested.

## Candidate layout and reports

`candidates/<role>/<fold>/` preserves the original portable exports. `robust` is primary; `strict` imposes tighter Gamma errors; `flexible`, `saturating` and `all_angle_diagnostic` are comparison candidates. Folds such as `omit_low`, `omit_high` and `omit_both` are full refits with endpoints removed. Consult the configuration and report before interpreting each role.

- [Complete report, 22 pages](reports/wse2_top3_gamma_complete.pdf)
- [All-angle bands](reports/all_angles_top3_focus_robust.pdf)
- [Gamma top2/top3 comparison](reports/gamma_top23_before_after.pdf)
- [Parameter functions](reports/parameter_functions.pdf)
- [Endpoint deletion](reports/endpoint_bands_both.pdf)
- [Small-angle sensitivity](reports/small_angle_top3_sensitivity.pdf)
- [Path bandwidths and gaps](reports/top3_path_bandwidth_gap.pdf)

The calibrated window is 2.13393–5.08585°. At larger angles the existing top3 mismatch remains; below 2.13° there is no independent own-Heff validation. Only top3 are fitted; the six plotted bands include diagnostic context. Reference curves are saved projected Heff results. Bandwidth/gap figures use the Gamma–M–K–Gamma path, not full-BZ extrema. No Chern number is inferred from the Gamma-only scan, and no old Chern result is transferred to these coefficients. Sensitivity spreads are not uncertainty bounds. See `reports/ORIGINAL_README.md`, `PROTOCOL.json`, `SUMMARY.json` and `VERIFICATION.json` for the full scope.

This is a single-particle model archive, not a JAQMC many-body training preset; uploading it does not implement the additional harmonics or momentum-dependent tunneling in the JAQMC estimator.
