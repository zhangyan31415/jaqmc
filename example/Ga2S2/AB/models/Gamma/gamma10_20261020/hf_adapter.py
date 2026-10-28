"""One-body HF matrix API, diagnostic candidate only; units eV and angstrom."""
from pathlib import Path
import importlib.util
_P=Path(__file__).resolve().parent
_s=importlib.util.spec_from_file_location('_local_gamma10',_P/'angle_model.py')
_M=importlib.util.module_from_spec(_s);_s.loader.exec_module(_M)
def hamiltonian(theta_deg,k_fractional):
    return _M.source_hamiltonian(theta_deg,k_fractional)
def reciprocal_basis(theta_deg):
    return _M._get().reciprocal(theta_deg)
def parameters(theta_deg):
    return _M.parameters(theta_deg)
