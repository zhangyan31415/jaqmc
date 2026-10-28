"""Single-particle electron matrix: HF fractional k uses reciprocal_basis()."""
import numpy as np
from angle_model import AngleModel
_MODEL=AngleModel()
def reciprocal_basis(theta_deg):
    return _MODEL.geometry(theta_deg)["reciprocal"].copy()
def fractional_to_model(theta_deg):
    """Archived band-path transformation; differs from the HF primitive basis."""
    return _MODEL.geometry(theta_deg)["fractional_to_model"].copy()
def hamiltonian(theta_deg,k_fractional,valley="K",nq=169):
    """HF fractional coordinates: k_cartesian=k_fractional@reciprocal_basis(theta)."""
    return _MODEL.hamiltonian(theta_deg,np.asarray(k_fractional)@reciprocal_basis(theta_deg),valley=valley,nq=nq)
def source_path_hamiltonian(theta_deg,k_source_fractional,valley="K",nq=169):
    """Archived plot fractional coordinates, transformed with source F instead of B."""
    return _MODEL.hamiltonian(theta_deg,np.asarray(k_source_fractional)@fractional_to_model(theta_deg),valley=valley,nq=nq)
def hole_hamiltonian(theta_deg,k_fractional,valley="K",nq=169):
    h=hamiltonian(theta_deg,-np.asarray(k_fractional),valley,nq)
    zero=hamiltonian(theta_deg,[0.,0.],valley,nq)
    return np.linalg.eigvalsh(zero)[-1]*np.eye(len(h))-h.T
