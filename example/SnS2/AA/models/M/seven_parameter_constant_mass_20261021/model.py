"""Seven local terms, actual layer axes, two constant kinetic eigenvalues."""
from pathlib import Path
import json,importlib.util
import numpy as np
_ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((_ROOT/'COEFFICIENTS.json').read_text());CATALOG=json.loads((_ROOT/'CATALOG.json').read_text());GEOMETRY=np.load(_ROOT/'geometry.npz')
def _module(name):
    spec=importlib.util.spec_from_file_location('_seven_'+name+str(_ROOT),_ROOT/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
_K=_module('kernel');_P=_module('physical')
def record(theta):
    r=next((r for r in CATALOG if abs(r['theta_deg']-float(theta))<.006),None)
    if r is None:raise ValueError('Hamiltonian geometry is certified only at twelve catalog angles')
    return r
def parameters(theta):
    s=np.sin(np.deg2rad(theta)/2)**2;lo,hi=np.sin(np.deg2rad(CONFIG['angle_endpoint_deg'])/2)**2;x=(s-lo)/(hi-lo);u=x/(CONFIG['rho']+(1-CONFIG['rho'])*x)
    return np.array(CONFIG['global_controls'])@np.array([(1-u)**2,2*u*(1-u),u*u])
def full_parameters(theta):
    p=parameters(theta);a=np.zeros((9,)+p.shape[1:]);a[CONFIG['active_full9_indices']]=p;return a
def geometry(theta):
    r=record(theta);return {n:GEOMETRY[f"{r['index']}_{n}"] for n in ['B','q1','q2','kappa1','kappa2']}
def hamiltonian(theta,k_fractional):
    g=geometry(theta);o=_K.operators(g,k_fractional,9,_P.m_geometry(g,k_fractional));return np.einsum('a,aij->ij',full_parameters(theta),o)
def realspace(theta,r):
    g=geometry(theta);return _K.potentials(r,g['B'],g['kappa1'],g['kappa2'],full_parameters(theta),9,_P.stars)
def kinetic_tensors(theta):
    g=geometry(theta);p=parameters(theta);kap=np.array([g['kappa1'],g['kappa2']]);ee=kap/np.linalg.norm(kap,axis=1)[:,None]
    return np.array([p[0]*np.outer(e,e)+p[1]*np.outer([-e[1],e[0]],[-e[1],e[0]]) for e in ee])
def enlarged_cell(theta):
    B=geometry(theta)['B'];return 2*np.pi*np.linalg.inv(np.array([(B[0]-B[1])/2,(B[0]+B[1])/2])).T
