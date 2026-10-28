from pathlib import Path
import json,socket,sys,numpy as np
from hf_adapter import *
from scipy.linalg import eigh
rows=[]
for r in RECORDS:
 theta=r['angle_deg'];g=geometry(theta);z=np.load(ROOT/'frozen'/(r['case']+'.npz'));ix=[0,17,20,40];vals=[];herm=0
 for i in ix:
  H=hamiltonian(theta,g['fractional'][i]);herm=max(herm,float(np.max(abs(H-H.conj().T))));vals.append(eigh(H,subset_by_index=[0,9],eigvals_only=True))
 # Fitted frozen output uses the Gamma first-pair mean as zero.
 vals=np.array(vals);vals-=vals[0,:2].mean();error=float(np.max(abs(vals-z['model_energy'][ix,:10])))
 assert herm<1e-12 and error<1e-9,(r['case'],herm,error)
 rows.append(dict(case=r['case'],hermiticity_max_eV=herm,frozen_spectrum_max_error_eV=error,indices=ix))
report=dict(status='PASS',node=socket.gethostname(),environment_prefix=sys.prefix,command=[sys.executable,*sys.argv],process_thread_layout='one process, BLAS up to4 threads',scope='portable one-body exactness only; no claim of scientific quality, native source reproduction, HF convergence or interaction validity',rows=rows)
(ROOT/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
