"""Uniform HF one-body API: electron conduction operator, eV."""
from pathlib import Path
import importlib.util
_p=Path(__file__).resolve().with_name('model.py')
_s=importlib.util.spec_from_file_location('portable_'+_p.parent.name,_p)
_m=importlib.util.module_from_spec(_s);_s.loader.exec_module(_m)
def hamiltonian(theta_deg,k_fractional=(0.,0.)):
    """Unshifted spatial H; no per-k or Gamma subtraction. Spin handled by caller."""
    return _m.hamiltonian(theta_deg,k_fractional)
def reciprocal_basis(theta_deg):
    """Rows B in inverse Angstrom; k_cartesian=k_fractional@B."""
    return _m.geometry(theta_deg)['B']
def plane_waves(theta_deg):
    """Cartesian Q in source order; two-component models use component-major ordering."""
    return _m.geometry(theta_deg)['q']
