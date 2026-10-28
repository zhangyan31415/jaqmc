# WSe2 AA K: current17-angle seven-term candidate

This is the current joint model, replacing neither data nor branch conventions with a symmetry average. Seven coefficients per angle are `a_h,Vcos,Vsin,w1,V2,w2,lambda`; five cubic Bernstein curves and one linear lambda curve, plus one shared mass, give23 global coefficients. `lambda` is a first-derivative interlayer operator and MUST be retained. The mass is0.35574179033919967 m_e.

Run from this directory with NumPy: `from hf_adapter import hamiltonian; H=hamiltonian(1.4701297257779093,[0,0])`. HF fractional coordinates use `k_cartesian=k_fractional@reciprocal_basis(theta)`. The archived band-path fractional coordinates instead use `fractional_to_model(theta)` and the separate `source_path_hamiltonian` helper. Native `AngleModel.hamiltonian` takes Cartesian k in inverse angstrom. Energies are eV. Reports compare model against projected Heff; overlap is raw same-rank squared overlap with separate group diagnostics, not native TAPW fidelity.

`K` contains one selected spin-valley branch. `Kprime` is its time-reversed conjugate at opposite momentum; do not multiply an additional free spin factor. Electron dispersion is negative quadratic; the optional hole conversion conjugates the basis and changes momentum. Use a consistent carrier/filling convention in HF. No interaction, form-factor, HF scan or NQS run is supplied by this archive.

The complete reciprocal and corresponding real-space formulas are in FORMULAS.md. `reports/COMPLETE_REPORT.pdf` contains bands, overlap and parameter curves. The large-angle wavefunction limitation remains: at5.09 degrees top3 RMS2.647meV, M-band3 overlap squared about0.440. Top4-6 were not fitted. Calibration is1.202855..5.085848 degrees.

`diagnostics/structure_response_rejected` preserves the later physically constrained angle-law/reallocation exploration, which FAILED its energy acceptance caps. It is report-only and is not a replacement model.

`CALIBRATION_PARAMETERS.csv` contains only this directory's physical parameters at every calibration angle. Original multi-branch report tables remain under `reports/`.
