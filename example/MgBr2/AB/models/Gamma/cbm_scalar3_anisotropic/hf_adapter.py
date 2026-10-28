from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parent
PARAM=json.loads((ROOT/'parameters.json').read_text());RECORDS=PARAM['records']
def geometry(theta_deg):
 if abs(theta_deg-RECORDS[0]['angle_deg'])>.005:raise ValueError('Only 3.890238 degrees was evaluated')
 return np.load(ROOT/'geometry/8_3.89.npz')
def reciprocal_basis(theta_deg):return geometry(theta_deg)['B'].copy()
def hamiltonian(theta_deg,k_fractional):
 g=geometry(theta_deg);q=g['q'];B=g['B'];A,Ca,Cb=RECORDS[0]['parameters'];H=np.diag(A*np.sum((q+np.asarray(k_fractional)@B)**2,axis=1)).astype(complex);diff=q[:,None]-q[None,:]
 for G,C in zip([B[0],B[1]-B[0],-B[1]],[Ca,Cb,Ca]):
  plus=(np.max(abs(diff-G),axis=2)<1e-7).astype(float);H+=C*(plus+plus.T)
 return H
ham=hamiltonian
