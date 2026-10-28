"""Verify portability against frozen source spectra, not a physics acceptance test."""
from pathlib import Path
import importlib.util,json,os,socket,numpy as np
ROOT=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_verified_hf_adapter',ROOT/'hf_adapter.py')
M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
info=json.loads((ROOT/'MODEL_INFO.json').read_text())
rows=[];herm=[]
if (ROOT/'FROZEN_BANDS.npz').exists():
 z=np.load(ROOT/'FROZEN_BANDS.npz');angles=z['angles_deg']
 for ai in [0,len(angles)//2,len(angles)-1]:
  theta=float(angles[ai]);pref=f'angle_{ai:02d}_';fr=z[pref+'fractional'];expected=z[pref+'model_energy'];origin=np.linalg.eigvalsh(M.hamiltonian(theta,fr[0]))[0]
  for ki in [0,13,20,40]:
   h=M.hamiltonian(theta,fr[ki]);herr=float(np.max(abs(h-h.conj().T)));herm.append(herr)
   actual=np.linalg.eigvalsh(h)[:6]-origin;want=expected[ki,:6]-expected[0,0]
   rows.append(dict(angle_deg=theta,k_index=ki,dimension=len(h),max_dispersion_error_eV=float(np.max(abs(actual-want)))))
else:
 z=np.load(ROOT/'FROZEN_SPECTRA.npz');theta=info['angles_deg'][0];B=M.reciprocal_basis(theta)
 for ki in [0,13,20,40,60]:
  fr=z['kpoints'][ki]@np.linalg.inv(B);h=M.hamiltonian(theta,fr);herm.append(float(np.max(abs(h-h.conj().T))))
  want=-z['electron_eigvals'][ki,::-1]
  rows.append(dict(angle_deg=theta,k_index=ki,dimension=len(h),max_absolute_spectrum_error_eV=float(np.max(abs(np.linalg.eigvalsh(h)-want)))))
maxerr=max(r.get('max_dispersion_error_eV',r.get('max_absolute_spectrum_error_eV')) for r in rows)
result=dict(state='PASS' if maxerr<1e-8 and max(herm)<1e-10 else 'FAIL',scope='Portable evaluator replay at representative stored k and angles; not validation of fitted physics, HF, Q convergence or native overlap.',max_spectrum_replay_error_eV=maxerr,max_Hermiticity_error_eV=max(herm),checks=rows,hostname=socket.gethostname(),cwd=str(Path.cwd()),environment_prefix=os.environ.get('CONDA_PREFIX'),python_executable=__import__('sys').executable,process_layout='one Python process; BLAS/OpenMP 4 threads maximum',command='python verify.py',returncode=0 if maxerr<1e-8 and max(herm)<1e-10 else 1)
(ROOT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(ROOT.name,result['state'],maxerr)
if result['state']!='PASS':raise SystemExit(1)
