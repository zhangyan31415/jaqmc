# MoTe₂ AA: six-parameter K-valley joint-fit example

This archive contains the selected 2026-10-16 `mixed` angle-function candidate, not a certified small-angle transferable material model. It preserves the existing fitted coefficients; no refit or new DFT calculation was performed for this upload.

## Model and scope

The original local two-layer, two-shell Hamiltonian has six coefficients: `a_h, Vcos, Vsin, w1, V2, w2`. Electron kinetics are purely quadratic, `-a_h |k-Q|²`. There is no beta or momentum-dependent tunneling. Energies are in eV, wavevectors in Å⁻¹, and `a_h` in eV Å². One mass, 0.6793823256145924 m_e, is shared by all nine calibration angles (2.13393–5.08585°; 2.28° was excluded from this fit).

With `r=sin(theta/2)/sin(3°/2)` and `z=r²/(1+r²)`, Vcos/Vsin use quadratics in z, w1/w2 use quadratics in r, and V2 uses a quadratic in r². There are 16 global fitted coefficients. `ANGLE_FUNCTION.json` stores the original controls and `angle_parameters.py` evaluates them. Dependence on r⁴ is an angle law for a potential, not a higher-order kinetic term.

`model.py` is the original portable evaluator; its leading docstring describes its four-parameter base, while `extended_response(..., family='second_harmonic')` implements the exported six-parameter family. `cases/<original-case-label>/model.npz` contains the actual geometry, family and six coefficients for each calibration angle. The two-decimal case labels are historical identifiers; exact angles are in `reports/REPORT.json`.

## Use

From this directory, with Python and NumPy:

```sh
python -c "from angle_parameters import AngleParameters; print(AngleParameters().physical_parameters(3.89))"
python -c "from model import SimplifiedModel; import numpy as np; m=SimplifiedModel('cases/8_3.89'); print(np.linalg.eigvalsh(m.hamiltonian([0.,0.]))[-3:][::-1])"
```

`SimplifiedModel.hamiltonian(k, carrier='electron', coordinates='model')` accepts Cartesian k in Å⁻¹. `coordinates='fractional'` uses the stored transformation. `carrier='hole'` uses the source's conjugate-basis hole conversion. Calibration-case geometry is supplied explicitly; the angle-parameter evaluator alone does not reconstruct a new small-angle geometry.

## Figures and limits

- [All-angle top3 bands and overlaps](reports/joint_top3_overlap.pdf)
- [Parameter trends](reports/joint_parameter_trends.pdf)
- [Delete-one-angle refits](reports/joint_leave_one_angle_out.pdf)
- [Small-angle sensitivity](reports/joint_small_angle_bands.pdf)
- [High-symmetry gaps](reports/joint_high_symmetry_gaps.pdf)

Reference curves in these comparisons are stored projected Heff bands, not additional independent DFT calculations. The selected candidate has known omitted-angle state-correspondence failures and unvalidated small-angle transferability; the complete limitations and numerical results are retained in `reports/REPORT.json`, `SMALL_ANGLE_REPORT.json` and `VERIFICATION.json`. Report figures focus on the historically fitted top3 bands and are archived unchanged. No new Chern calculation or JAQMC many-body training is claimed. This example is a portable single-particle model archive, not a ready-to-run JAQMC training preset.
