# MgBr2 AB Gamma conduction scalar three-parameter ablation

**DIAGNOSTIC_ONLY; HF default disabled.** Different conduction target from the original valence problem. The API returns one scalar spatial block in eV, with electron energy convention. A spin-independent approximation can duplicate it using `np.kron(H,np.eye(2))`; those spectator spins do not reproduce the tiny native SOC splitting. See METHODS and REPORT for the fixed-source origin and density limitations.

```python
from hf_adapter import hamiltonian, reciprocal_basis
H = hamiltonian(3.8902381690076835, [0.13, 0.07])
```

Single-angle parameter removal from a five-parameter pilot, not a new optimized three-parameter fit. First two raw pair overlap minima are 0.9574 and0.9562 but K gap is 0.0823 meV versus native0.1591 meV. No all-angle curves, atomic wavefunctions, H/S, or microscopic interaction form factors are bundled. `python verify.py` checks Hermiticity and frozen model spectra, never the original atomic eigensystem.
