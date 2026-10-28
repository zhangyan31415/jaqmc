import json,socket,sys,numpy as np
from scipy.linalg import eigh
from hf_adapter import *
theta=RECORDS[0]['angle_deg'];g=geometry(theta);z=np.load(ROOT/'frozen/8_3.89.npz');ix=[0,17,20,40];H=[hamiltonian(theta,g['fractional'][i])for i in ix];herm=max(float(np.max(abs(h-h.conj().T)))for h in H);E=np.array([np.repeat(eigh(h,eigvals_only=True)[:5],2)for h in H]);E-=E[0,0];err=float(np.max(abs(E-z['model_energy'][ix,:10])));assert herm<1e-12 and err<1e-9
r=dict(status='PASS',node=socket.gethostname(),environment_prefix=sys.prefix,command=[sys.executable,*sys.argv],process_thread_layout='one process BLAS<=4',scope='one-body export exactness; not fit or HF validation',hermiticity_max_eV=herm,frozen_spectrum_max_error_eV=err,indices=ix);(ROOT/'VERIFICATION.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
