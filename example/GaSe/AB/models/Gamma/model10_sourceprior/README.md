# GaSe AB Gamma / model10_sourceprior

Small-angle first-two-conduction-band exploratory candidate; all-angle three-band accuracy fails.

This is a portable **one-body continuum Hamiltonian**, not an HF run configuration. 10 physical coefficients per angle; 26 independent global angle coefficients. Mass coefficient(s) A1,A2 are constant across the 12 jointly fitted angles. All energies are eV; lengths Angstrom.

```python
import importlib.util
from pathlib import Path
p = Path("model.py").resolve()  # run from this package directory
spec = importlib.util.spec_from_file_location("continuum_model", p)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
angle = m.INFO["angles_deg"][0]
parameters = m.parameters(angle)
g = m.geometry(angle)
H = m.hamiltonian(angle, k_fractional=[0.0, 0.0])
V = m.potential([[0.0, 0.0]], g["B"], parameters)
```

`parameters(theta)` evaluates the common quadratic Bernstein law. Write s=sin(theta/2)^2, x=(s-smin)/(smax-smin), u=x/[rho+(1-rho)x], rho=0.2; p=c0(1-u)^2+2c1u(1-u)+c2u^2. Geometries use the exact stored source Q/B at each training angle. For a custom converged Q cutoff, pass Cartesian `q` and `B` into `hamiltonian`; the package does not extrapolate geometry or revalidate physics. `hamiltonian` returns only spatial channels; physical spectator spin duplicates its spectrum, and spin/valley flavors must be handled explicitly in HF.

H_ll=-A_l laplacian+Ue+eta_l Uo; H_12=H_21=t0+Tc*C1+Ts*S1. Ue=Ve_c*C1+Ve_s*S1 (+V2c*C2 for 10p); Uo=Vo_c*C1+Vo_s*S1.

C_s=2 sum_j cos(G_sj.r), S_s=2 sum_j sin(G_sj.r). With B rows a,b, stars are [a,b-a,-b], [a+b,b-2a,a-2b], and 2*[a,b-a,-b]. Fourier transfer q-q'=+G has coefficient Vc-iVs, -G has Vc+iVs. Kinetic term is A_l*|q+k_fractional@B|^2. Two-component models use component-major indexing. The energy comparisons use one common Gamma-lowest-level offset, not separate per-band shifts.

Parameter order: `A1_eV_A2, A2_eV_A2, t0_eV, Ve_c_eV, Ve_s_eV, Vo_c_eV, Vo_s_eV, Tc_eV, Ts_eV, V2c_eV`. Full fixed-origin conventions, fitting loss, capture/training masks and native-reference provenance are in `reports/SOURCE_METHOD.md` and `reports/RESULT.json`. `data/*.npz` contain only Q/B/path geometry and small saved energies/probabilities, not raw Hamiltonians or eigenvectors. Native mean pair probability and minimum principal probability measure different things; model-degenerate cluster coverage is not a resolved-band overlap. The PNG legends mark these cases.

Limitations:

- No candidate passes all-angle three-band energy/overlap requirements.
- Scalar spectator-spin model; native SOC splitting is not reproduced.
- Two-channel source layer-weight bias remains.
- 4/6 parameter high bands above 4.41 degrees have restricted source capacity and training masks.
- No HF solver or interaction projection is supplied. Physical carrier spin and valley flavor multiplicity must be specified separately by the HF implementation.
- Saved native overlaps are absolute probabilities, never divided by source capture; Pmean, principal probability and whole degenerate-cluster coverage are distinct.
- Finite model Q expansion checks do not certify the original database TAPW Q convergence.
- Absolute band-edge offsets are removed by a common Gamma lowest-pair origin; do not infer inter-valley energy alignment from this model.
- Angle law is valid only inside the jointly fitted interval; stored geometry is available only at 12 training angles.

Run `python verify.py` in this directory to check portability, saved spectra, Hermiticity and the numerical real/Fourier transform. `VERIFICATION.json` records the fresh export check; `reports/VERIFICATION.json`, when present, is the original scientific-fit check. Passing export verification does not qualify a failed physical model.

For the uniform HF exporter, `hf_adapter.py` exposes `hamiltonian(theta_deg, k_fractional)`, `reciprocal_basis(theta_deg)`, and `plane_waves(theta_deg)`. H is the unshifted electron conduction operator, without subtracting Gamma energy; only report spectra use that common offset. The matrix follows stored Q order; two-channel models use all Q of channel 1 followed by all Q of channel 2.
