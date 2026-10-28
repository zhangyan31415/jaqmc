"""Single-angle spectral diagnostic. Returned H is the HOLE Hamiltonian -H_e."""
from pathlib import Path
import importlib.util
import numpy as np
_ROOT=Path(__file__).resolve().parent
_s=importlib.util.spec_from_file_location('_local_spectral_WTe2',_ROOT/'physical_model.py')
_P=importlib.util.module_from_spec(_s);_s.loader.exec_module(_P)
_MODEL=_P.load_model(_ROOT)
THETA=3.8902381690076835
def _check(theta_deg):
    if abs(float(theta_deg)-THETA)>0.006:raise ValueError('Only the archived single angle is fitted.')
def reciprocal_basis(theta_deg):
    _check(theta_deg)
    return _MODEL._basis.copy()
def electron_hamiltonian(theta_deg,k_fractional):
    _check(theta_deg)
    h=_MODEL.hamiltonian(np.asarray(k_fractional)@_MODEL._basis)
    # Export's eigenvalue evaluator explicitly Hermitianizes before diagonalization.
    return (h+h.conj().T)/2
def hamiltonian(theta_deg,k_fractional):
    return -electron_hamiltonian(theta_deg,k_fractional)
