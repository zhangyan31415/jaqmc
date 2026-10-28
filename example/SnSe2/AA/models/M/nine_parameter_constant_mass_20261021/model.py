"""Self-contained spin-spectator electron M-valley Hamiltonian, eV and Angstrom."""
from pathlib import Path
import importlib.util,json
import numpy as np
_ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((_ROOT/'COEFFICIENTS.json').read_text())
CATALOG=json.loads((_ROOT/'CATALOG.json').read_text())
GEOMETRY=np.load(_ROOT/'geometry.npz')
def _module(name):
 s=importlib.util.spec_from_file_location('_mvalley_'+name+str(_ROOT),_ROOT/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
_K=_module('kernel');_P=_module('physical')
def record(theta):
 r=next((r for r in CATALOG if abs(r['theta_deg']-float(theta))<.006),None)
 if r is None:raise ValueError('Geometry is supplied only at the catalog angles')
 return r
def parameters(theta):
 t=np.asarray(theta);lo,hi=CONFIG['angle_endpoint_deg']
 if np.any(t<lo-1e-8) or np.any(t>hi+1e-8):raise ValueError('Outside fitted angle interval; use record(label)["theta_deg"]')
 s=np.sin(np.deg2rad(t)/2)**2;s0,s1=np.sin(np.deg2rad([lo,hi])/2)**2;x=(s-s0)/(s1-s0);u=x/(CONFIG['rho']+(1-CONFIG['rho'])*x)
 return np.stack([(1-u)**2,2*u*(1-u),u*u],axis=-1)@np.asarray(CONFIG['global_controls']).T
def geometry(theta):
 r=record(theta);return {n:GEOMETRY[f"{r['index']}_{n}"] for n in ['B','q1','q2','kappa1','kappa2']}
def operators(theta,k_fractional):
 g=geometry(theta);ops=_K.operators(g,k_fractional,9,_P.m_geometry(g,k_fractional))
 if CONFIG['physical_parameter_count']==9:return ops
 axis,obs=_P.stars(g['B'],g['kappa1']-g['kappa2'],g['kappa1']+g['kappa2']);n1=len(g['q1']);extra=np.zeros(ops.shape[1:],complex)
 for offset,q in [(0,g['q1']),(n1,g['q2'])]:
  d=q[None]-q[:,None];a=sum((np.max(abs(d-v),axis=-1)<1e-7).astype(float)+(np.max(abs(d+v),axis=-1)<1e-7).astype(float) for v in obs);extra[offset:offset+len(q),offset:offset+len(q)]=a
 return np.concatenate([ops,extra[None]])
def hamiltonian(theta,k_fractional):return np.einsum('a,aij->ij',parameters(theta),operators(theta,k_fractional))
def realspace(theta,r):
 g=geometry(theta);p=parameters(theta);v=_K.potentials(r,g['B'],g['kappa1'],g['kappa2'],p[:9],9,_P.stars)
 if len(p)==10:
  _,obs=_P.stars(g['B'],g['kappa1']-g['kappa2'],g['kappa1']+g['kappa2']);v[...,:2]+=2*p[9]*np.cos(np.asarray(r)@obs.T).sum(axis=-1)[...,None]
 return v

def kinetic_tensors(theta):
 g=geometry(theta);p=parameters(theta);e=np.array([g['kappa1'],g['kappa2']]);e/=np.linalg.norm(e,axis=1)[:,None];return p[1]*np.eye(2)[None]+(p[0]-p[1])*e[:,:,None]*e[:,None,:]
def enlarged_cell(theta):
 B=geometry(theta)['B'];Q=np.array([(B[0]-B[1])/2,(B[0]+B[1])/2]);return 2*np.pi*np.linalg.inv(Q).T
