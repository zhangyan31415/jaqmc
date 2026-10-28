"""Electron one-body matrix, eV; fractional k is with respect to B rows."""
from pathlib import Path
import importlib.util
import numpy as np
_path=Path(__file__).resolve().with_name("model.py")
_spec=importlib.util.spec_from_file_location("_historical_m_model",_path)
_model=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_model)
def hamiltonian(theta_deg,k_fractional):
    return _model.hamiltonian(theta_deg,np.asarray(k_fractional))
def reciprocal_basis(theta_deg):
    record=_model.record(theta_deg)
    return _model.geometry(record["theta_deg"])["B"].copy()
