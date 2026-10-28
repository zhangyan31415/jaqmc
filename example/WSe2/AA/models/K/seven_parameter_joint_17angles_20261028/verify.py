"""Reload standalone models and compare unchanged frozen source spectra; no fit/HF run."""
import json,sys,os,socket,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.linalg import eigh
from hf_adapter import hamiltonian,reciprocal_basis,fractional_to_model,hole_hamiltonian
from angle_model import AngleModel
m=AngleModel();info=json.loads((ROOT/'MODEL_INFO.json').read_text());source=json.loads((ROOT/'FROZEN_REFERENCE.json').read_text())
records=[];eh=0.;et=0.;ep=0.;es=0.
for row in source['records']:
    theta=row['theta_deg'];source_k=np.array(row['k_fractional']);k=source_k@fractional_to_model(theta)@np.linalg.inv(reciprocal_basis(theta));h=hamiltonian(theta,k);n=len(h)
    herm=float(np.max(abs(h-h.conj().T)));eh=max(eh,herm)
    energy=eigh(h,eigvals_only=True,subset_by_index=[n-6,n-1]);delta=float(np.max(abs(energy-np.array(row['top6_eV']))));es=max(es,delta)
    if 'parameters' in row:par=np.array(row['parameters'])
    else:
        with np.load(ROOT/'cases'/row['case']/'model.npz') as z:par=z['parameters']
    dp=float(np.max(abs(m.parameters(theta)-par)));ep=max(ep,dp)
    tr=float(np.max(abs(hamiltonian(theta,k,valley='Kprime')-hamiltonian(theta,-k).conj())));et=max(et,tr)
    assert np.isfinite(h).all() and np.all(np.isfinite(reciprocal_basis(theta)))
    records.append(dict(theta_deg=theta,k_fractional_HF=k.tolist(),k_fractional_source=source_k.tolist(),n=n,source_top6_max_error_eV=delta,Hermitian_max_eV=herm,time_reversal_max_eV=tr,parameter_max_error=dp))
theta=info['angles_deg'][0];k=np.array([.13,-.07]);h=hamiltonian(theta,-k);zero=hamiltonian(theta,[0.,0.]);expected=np.linalg.eigvalsh(zero)[-1]*np.eye(len(h))-h.T;dh=float(np.max(abs(hole_hamiltonian(theta,k)-expected)))
assert es<1e-9,(es,info['id']);assert eh<1e-11;assert et<1e-11;assert ep<1e-11;assert dh<1e-11
# Explicitly forbid this package from importing any implementation from the source moirekp tree.
imports={name:str(getattr(module,'__file__','')) for name,module in list(sys.modules.items()) if getattr(module,'__file__',None)}
assert not any('1.tapw_code/moirekp-release' in value for value in imports.values())
record=dict(status='passed',scope='Fresh standalone imports, all calibration angles at3-4path points; frozen source top6 absolute eigenvalues, parameter identity, Hermiticity, K/Kprime time reversal, hole conversion. This is not an HF calculation.',hostname=socket.gethostname(),cwd=str(Path.cwd()),environment_prefix=sys.prefix,exact_command=[sys.executable,'-I',str(Path(__file__).resolve())],process_thread_layout='one process;4 BLAS/OpenMP threads; sequential packages',reference=source['reference'],reference_sha256=hashlib.sha256((ROOT/'FROZEN_REFERENCE.json').read_bytes()).hexdigest(),reference_records=len(records),max_source_top6_error_eV=es,max_parameter_error=ep,max_Hermitian_error_eV=eh,max_time_reversal_error_eV=et,hole_conversion_error_eV=dh,source_tree_imports=0,checks=records,local_module_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},finished_unix=time.time(),exit_code=0)
(ROOT/'VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\n')
print(info['id'],'PASSED',len(records),'source spectrum max eV',es,'Hermitian',eh,'TR',et,flush=True)
