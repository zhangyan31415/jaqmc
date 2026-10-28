"""Portable continuum candidate. See COEFFICIENTS.json for accuracy and locality."""
from pathlib import Path
import json,importlib.util
import numpy as np
_D=Path(__file__).resolve().parent
_CFG=json.loads((_D/'COEFFICIENTS.json').read_text())
_SRC='physical_soc.py' if _CFG['family']=='gamma' and _CFG['physical_parameter_count']==9 else 'physical.py'
_S=importlib.util.spec_from_file_location('_expanded_physical',_D/_SRC)
_P=importlib.util.module_from_spec(_S);_S.loader.exec_module(_P)
CONFIG=json.loads((_D/'COEFFICIENTS.json').read_text())
CATALOG=json.loads((_D/'CATALOG.json').read_text())
GEOMETRY=np.load(_D/'geometry.npz')
def record(theta):
    rec=next((a for a in CATALOG if abs(a['theta_deg']-float(theta))<.006),None)
    if rec is None:raise ValueError('Use a catalog angle; no off-grid accuracy certificate.')
    return rec
def parameters(theta):
    theta=record(theta)['theta_deg'];lo,hi=np.sin(np.deg2rad(CONFIG['angle_endpoint_deg'])/2)**2
    s=np.sin(np.deg2rad(theta)/2)**2
    u=(1/s-1/lo)/(1/hi-1/lo) if CONFIG.get('angle_coordinate')=='inverse_sin2' else (s-lo)/(hi-lo)
    if CONFIG.get('angle_coordinate')=='warped_sin2':u=u/(CONFIG['rho']+(1-CONFIG['rho'])*u)
    b=np.array([1-u,u]) if CONFIG['angle_degree']==1 else np.array([(1-u)**2,2*u*(1-u),u*u])
    return np.array(CONFIG['global_controls'])@b
def geometry(theta):
    rec=record(theta);names=['q','B'] if CONFIG['family']=='gamma' else ['B','q1','q2','kappa1','kappa2']
    return {name:GEOMETRY[f"{rec['index']}_{name}"] for name in names}
def hamiltonian(theta,k_fractional):
    g=geometry(theta);p=parameters(theta)
    return _P.gamma_hj(g['q'],g['B'],k_fractional,p)[0] if CONFIG['family']=='gamma' else _P.m_hj(g,k_fractional,p,CONFIG['config'])[0]
def gamma_potentials(theta,r,xp=np):
    if CONFIG['family']!='gamma':raise ValueError('Gamma interface only')
    p=parameters(theta);B=geometry(theta)['B'];star=xp.asarray(np.array([B[0],B[1]-B[0],-B[1]]));a=xp.asarray(r)@star.T;c=2*xp.cos(a).sum(axis=-1);s=2*xp.sin(a).sum(axis=-1)
    return xp.stack([p[3]*c+p[4]*s,p[2]+p[5]*c+p[6]*s],axis=-1)
