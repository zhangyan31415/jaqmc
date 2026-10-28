"""Portable coefficient, Hermiticity, spectra and Fourier verification (no fit)."""
from pathlib import Path
import os,json,socket,sys,time
import numpy as np
import model

def run():
    start=time.time();rows=[];max_parameter_error=0.;max_hermitian_error=0.;max_spectrum_error=0.
    for item in model.DATA['cases']:
        angle=item['angle_deg'];p=model.parameters(angle);max_parameter_error=max(max_parameter_error,float(np.max(abs(p-item['parameters']))))
        g=model.geometry(angle)
        with np.load(model.ROOT/'data'/(item['case']+'.npz')) as z:expected=z['model_pair_centers']
        e0=np.linalg.eigvalsh(model.hamiltonian(angle,[0,0]))[0]
        errors=[]
        for i in [0,10,20,30,40,50,60]:
            H=model.hamiltonian(angle,g['fractional'][i]);max_hermitian_error=max(max_hermitian_error,float(np.max(abs(H-H.conj().T))))
            e=np.linalg.eigvalsh(H)[:3]-e0;errors.append(float(np.max(abs(e-expected[i]))))
        max_spectrum_error=max(max_spectrum_error,max(errors));rows.append(dict(case=item['case'],angle_deg=angle,k_indices=[0,10,20,30,40,50,60],max_spectrum_error_eV=max(errors)))
    # Numerical Fourier integral on an independent finite reciprocal grid.
    B=np.array([[.08,0],[.04,.04*np.sqrt(3)]])
    q=np.array([i*B[0]+j*B[1] for i in [-1,0,1] for j in [-1,0,1]])
    frac=np.array(np.meshgrid(np.arange(32)/32,np.arange(32)/32,indexing='ij')).reshape(2,-1).T
    r=frac@(2*np.pi*np.linalg.inv(B).T);phase=np.exp(-1j*q@r.T)
    p=model.parameters(model.DATA['cases'][0]['angle_deg']).copy();p[:2 if model.DATA['kind']=='layer' else 1]=0
    V=model.potential(r,B,p);o=model.operators(q,B);analytic=np.einsum('a,aij->ij',p,o)
    if V.ndim==1:numeric=(phase*V)@phase.conj().T/len(r)
    else:
        numeric=np.zeros_like(analytic);n=len(q)
        for l in [0,1]:
            for m in [0,1]:numeric[l*n:(l+1)*n,m*n:(m+1)*n]=(phase*V[:,l,m])@phase.conj().T/len(r)
    fourier_error=float(np.max(abs(numeric-analytic)))
    assert max_parameter_error<1e-10,max_parameter_error
    assert max_hermitian_error<1e-12,max_hermitian_error
    assert max_spectrum_error<1e-9,max_spectrum_error
    assert fourier_error<1e-12,fourier_error
    result=dict(status='PASSED',scope='Portable one-body export only: parameters at all 12 training angles; Hermiticity and first-three spectra at 7 path points per angle; independent numerical real/Fourier potential transform. No new fits, HF solver, interactions, or physical quality acceptance.',maximum_parameter_error=max_parameter_error,maximum_Hermitian_error_eV=max_hermitian_error,maximum_saved_spectrum_error_eV=max_spectrum_error,maximum_real_reciprocal_Fourier_error_eV=fourier_error,cases=rows,node=socket.gethostname(),cwd=str(Path.cwd()),environment_prefix=sys.prefix,command=[sys.executable,*sys.argv],layout='1 process; OMP/MKL/OpenBLAS/NumExpr threads=4',source_reference='data/*.npz saved model predictions copied from source RESULT/predictions',elapsed_seconds=time.time()-start,final_process_state='EXITED_SUCCESSFULLY')
    (model.ROOT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(model.INFO['id'],result['status'],max_spectrum_error,fourier_error,flush=True)
if __name__=='__main__':run()
