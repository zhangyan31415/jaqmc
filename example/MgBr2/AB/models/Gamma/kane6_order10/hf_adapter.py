"""Portable one-body diagnostic; no JAQMC training or interaction projection implied."""
from pathlib import Path
import json
import numpy as np
try:
 from . import operators as op
except ImportError:
 import operators as op
ROOT=Path(__file__).resolve().parent
PARAM=json.loads((ROOT/'parameters.json').read_text())
RECORDS=PARAM['records']
def record(theta_deg):
 r=min(RECORDS,key=lambda x:abs(x['angle_deg']-theta_deg))
 if abs(r['angle_deg']-theta_deg)>0.005:raise ValueError('Only tabulated validated geometries are available; no extrapolation')
 return r
def geometry(theta_deg):
 return np.load(ROOT/'geometry'/(record(theta_deg)['case']+'.npz'))
def reciprocal_basis(theta_deg):return geometry(theta_deg)['B'].copy()
def parameters(theta_deg):
 return np.array(record(theta_deg)['parameters'])
def hamiltonian(theta_deg,k_fractional):
 g=geometry(theta_deg);p=parameters(theta_deg)
 O=op.all_operators(g['q'],g['B'],k_fractional);ix=PARAM['potential_indices'];return np.einsum('a,aij->ij',p,np.concatenate([O[:8],O[8+np.array(ix)]],axis=0))

ham=hamiltonian
