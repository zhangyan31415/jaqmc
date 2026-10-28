# WS2 single-angle general spectral diagnostic

Actual NumPy-only matrix evaluator, explicit polynomial terms, Q geometry, coefficients and original comparison figures. There are **281 independent fitted real coefficients**, not ten or fewer. It is excluded from default HF scans.

Use `hf_adapter.hamiltonian(theta_deg, k_fractional)` for the hole matrix in eV; `electron_hamiltonian` retains the source electron convention. The adapter converts fractional coordinates and performs the same Hermitianization as the source spectral evaluator. `reciprocal_basis` gives rows in angstrom^-1. Read `MODEL_INFO.json` for restrictions. `MODEL.md` supplies the full position/momentum operator expansion.

The high-quality source projection in `SOURCE_PROJECTION_REPORT.md` is a separate object. Its acceptance does not transfer to this polynomial fit. This archive contains no atomic H/S and does not recompute native overlap.
