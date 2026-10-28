"""Portable one-body Gamma continuum, eV/Angstrom. No source-project dependency."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parent
INFO=json.loads((ROOT/'MODEL_INFO.json').read_text())
DATA=json.loads((ROOT/'data'/'coefficients.json').read_text())

def parameters(angle_deg):
    """Physical coefficients from the jointly fitted angle law; no extrapolation."""
    lo,hi=DATA['angle_endpoint_deg']
    if not lo-1e-9<=angle_deg<=hi+1e-9: raise ValueError('Angle outside joint-fit interval')
    s=np.sin(np.deg2rad([lo,hi,angle_deg])/2)**2
    x=(s[2]-s[0])/(s[1]-s[0]); u=x/(DATA['rho']+(1-DATA['rho'])*x)
    return np.array(DATA['global_controls'])@np.array([(1-u)**2,2*u*(1-u),u*u])

def geometry(angle_deg):
    """Exact stored source Q and reciprocal B at an available training angle."""
    item=min(DATA['cases'],key=lambda c:abs(c['angle_deg']-angle_deg))
    if abs(item['angle_deg']-angle_deg)>1e-7: raise ValueError('Geometry is stored only at the listed training angles')
    with np.load(ROOT/'data'/(item['case']+'.npz')) as z:
        return {key:z[key].copy() for key in ['q','B','fractional']}

def stars(B):
    a,b=np.asarray(B); g=np.array([a,b-a,-b])
    return [g,np.array([a+b,b-2*a,a-2*b]),2*g]

def operators(q,B,k_fractional=(0.,0.)):
    """H=sum p[a] operators[a]. Component-major basis for two-channel variants."""
    q=np.asarray(q);B=np.asarray(B);n=len(q);npar=INFO['physical_parameters'];kind=DATA['kind']
    p2=np.sum((q+np.asarray(k_fractional)@B)**2,axis=1);d=q[:,None]-q[None,:]
    cs=[];ss=[]
    for star in stars(B):
        plus=sum((np.max(abs(d-g),axis=2)<1e-7).astype(float) for g in star)
        cs.append(plus+plus.T);ss.append(-1j*(plus-plus.T))
    if kind=='layer':
        o=np.zeros((npar,2*n,2*n),complex);o[0,:n,:n]=np.diag(p2);o[1,n:,n:]=np.diag(p2)
        o[2,:n,n:]=np.eye(n);o[2,n:,:n]=np.eye(n)
        for l in [0,1]:
            sl=slice(l*n,(l+1)*n);eta=1 if l==0 else -1
            o[3,sl,sl]=cs[0];o[4,sl,sl]=ss[0];o[5,sl,sl]=eta*cs[0];o[6,sl,sl]=eta*ss[0]
        o[7,:n,n:]=cs[0];o[7,n:,:n]=cs[0];o[8,:n,n:]=ss[0];o[8,n:,:n]=ss[0]
        if npar==10:o[9,:n,:n]=cs[1];o[9,n:,n:]=cs[1]
        return o
    full=np.array([np.diag(p2),cs[0],ss[0],cs[1],ss[1],cs[2],ss[2]],complex)
    return full[:npar] if kind=='scalar' else full[[0,1,2,3] if npar==4 else [0,1,2,3,5,6]]

def hamiltonian(angle_deg,k_fractional=(0.,0.),q=None,B=None):
    """Return spinless spatial H in eV; physical spectator spin duplicates its spectrum."""
    if q is None or B is None:
        g=geometry(angle_deg);q=g['q'];B=g['B']
    return np.einsum('a,aij->ij',parameters(angle_deg),operators(q,B,k_fractional))

def potential(r_angstrom,B,p):
    """Local scalar or 2x2 potential. Real-space Hamiltonian is -A_l laplacian + V."""
    r=np.asarray(r_angstrom);p=np.asarray(p);shape=r.shape[:-1];c=[];s=[]
    for star in stars(B):
        a=r@star.T;c.append(2*np.cos(a).sum(axis=-1));s.append(2*np.sin(a).sum(axis=-1))
    if DATA['kind']=='layer':
        even=p[3]*c[0]+p[4]*s[0]+(p[9]*c[1] if len(p)==10 else 0)
        odd=p[5]*c[0]+p[6]*s[0];t=p[2]+p[7]*c[0]+p[8]*s[0]
        out=np.zeros(shape+(2,2),complex);out[...,0,0]=even+odd;out[...,1,1]=even-odd;out[...,0,1]=t;out[...,1,0]=t;return out
    terms=[c[0],s[0],c[1],s[1],c[2],s[2]]
    indices=list(range(len(p)-1)) if DATA['kind']=='scalar' else ([0,1,2] if len(p)==4 else [0,1,2,4,5])
    return sum(p[i+1]*terms[j] for i,j in enumerate(indices))
