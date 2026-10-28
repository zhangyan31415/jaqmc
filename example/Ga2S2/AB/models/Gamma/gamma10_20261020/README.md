# Ga2S2 Gamma 10-parameter diagnostic

This archive contains the actual one-body Hamiltonian, geometry, angle coefficients, energy/overlap figures and formula. It is **excluded from the default HF scan** because its native wavefunction gates failed. The runtime is portable (Python 3.11 + NumPy) and needs no original TAPW data or absolute imports.

`hf_adapter.hamiltonian(theta_deg, k_fractional)` returns a Hermitian electron Hamiltonian in eV at a catalog angle. `reciprocal_basis(theta_deg)` returns reciprocal rows in angstrom^-1. This supplies one-body matrices, not an HF solver or two-body interaction/form-factor prescription. The square-root kinetic energy must be retained; substituting a Laplacian changes the model. See `MODEL_INFO.json`, `FORMULA.txt`, `VERIFICATION.json` and `SOURCE_SHA256.json`.

No source atomic H/S, full native eigenvectors, or fitted-gauge normalization are bundled. The overlap plots are frozen diagnostics; the archive cannot recompute native overlaps without the source data.
