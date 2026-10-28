"""One-body HF interchange API; no interaction or HF solver is supplied."""
from pathlib import Path
import importlib.util
import numpy as np
_s=importlib.util.spec_from_file_location('_local_model_'+str(Path(__file__).parent),Path(__file__).with_name('model.py'));model=importlib.util.module_from_spec(_s);_s.loader.exec_module(model)
def hamiltonian(theta_deg,k_fractional):return model.hamiltonian(model.record(theta_deg)['theta_deg'],k_fractional)
def reciprocal_basis(theta_deg):return model.geometry(theta_deg)['B'].copy()
def basis_metadata(theta_deg):
 g=model.geometry(theta_deg);n1=len(g['q1']);n2=len(g['q2']);return dict(q=np.concatenate([g['q1'],g['q2']]),component=np.repeat([0,1],[n1,n2]),plane_wave='exp(i*(k_cart-q) dot r)',spin='spectator; duplicate H with I_spin if wanted',density_vertex='Envelope density only: equal component and q_out=q_in-Q for exp(i Q dot r); atomic TAPW density form factors not included')
