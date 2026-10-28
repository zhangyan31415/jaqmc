# MgBr2 kane6_order10: diagnostic archive

**DIAGNOSTIC_ONLY; excluded from HF default screening.** This archive is uploaded for reproducibility and explicit opt-in experiments. The one-body fit failed its wavefunction gate. It is not a validated JAQMC training preset.

```python
import sys
sys.path.insert(0, "/path/to/example/MgBr2/AB/models/Gamma/kane6_order10")
from hf_adapter import hamiltonian, reciprocal_basis
H = hamiltonian(3.8902381690076835, [0.13, 0.07])  # eV
B = reciprocal_basis(3.8902381690076835)  # row vectors, angstrom^-1
```

Dependencies: NumPy and SciPy. `geometry/` contains only small model Q and reciprocal geometry; no atomic H/S, native wavefunctions, or source lifts are bundled. `frozen/` stores model spectra and native-comparison scalar overlap diagnostics. `operators.py` retains every spinful matrix coupling; no scalarization. The effective Kramers label cannot be identified with laboratory spin or a physical layer. Density rho=identity in envelope channels is a separate approximation; microscopic density operators, projected interactions, and spin observables require source form factors not present here. HF users must supply and label those choices. Hole energy is minus electron energy; electron occupations cannot be reused without particle-hole conversion.

P is mean pair projector probability; PP is the smallest squared singular value for the pair. Both compare directly to actual native TAPW states using the source lift and are never divided by source capture. Native pairs are energy ordered, with unresolved model pair boundaries masked (MgBr2 replacement: third-pair crossing at k index51); cluster coverage is a separate diagnostic. Shown fit metrics are not guaranteed all-BZ bounds. See RESULTS/REPORT, Q checks, and METHODS for errors, exclusions, and conventions. MgBr2 is a single-angle pilot; no parameter-angle curve exists. PtSe2 has all twelve angles jointly fitted, with the two kinetic coefficients held constant.

Run `python verify.py` for portable Hermiticity and frozen-spectrum verification; it never refits parameters or diagonalizes the atomic problem.
