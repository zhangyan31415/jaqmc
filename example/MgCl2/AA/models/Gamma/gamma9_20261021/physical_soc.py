"""One additional Pauli SOC term for the Gamma curvature diagnostic."""
import numpy as np
import importlib.util
from pathlib import Path
_s=importlib.util.spec_from_file_location("_portable_base",Path(__file__).resolve().parent/"physical.py")
base=importlib.util.module_from_spec(_s);_s.loader.exec_module(base)
globals().update({k:v for k,v in vars(base).items() if not k.startswith("_")})
GAMMA_NAMES=base.GAMMA_NAMES+['lambda_A2']


def gamma_geometry(q,B,k):
    g=base.gamma_geometry(q,B,k);q=np.asarray(q);dq=q[:,None]-q[None,:];avg=(q[:,None]+q[None,:])/2+np.asarray(k)@B
    cross=dq[:,:,0]*avg[:,:,1]-dq[:,:,1]*avg[:,:,0]
    return (*g,cross)


def gamma_from_geometry(g,p):
    if len(p)!=9:return base.gamma_from_geometry(g[:3],p)
    h,j=base.gamma_from_geometry(g[:3],p[:8]);n=len(g[0]);c,s,cross=g[1:];soc=np.zeros_like(h);responses=[]
    for branch in range(2):
        sl=slice(branch*n,(branch+1)*n);pot=p[3+2*branch]*c+p[4+2*branch]*s;soc[sl,sl]=1j*pot*cross
    result=np.kron(h,np.eye(2))+p[8]*np.kron(soc,base.SZ);jac=np.array([np.kron(a,np.eye(2)) for a in j]+[np.kron(soc,base.SZ)])
    for branch in range(2):
        sl=slice(branch*n,(branch+1)*n)
        for index,op in [(3+2*branch,c),(4+2*branch,s)]:
            a=np.zeros_like(h);a[sl,sl]=1j*op*cross;jac[index]+=p[8]*np.kron(a,base.SZ)
    return result,jac


def gamma_hj(q,B,k,p):return gamma_from_geometry(gamma_geometry(q,B,k),p)


def spin_embedding(n):
    u=np.zeros((4*n,4*n),complex);q=np.arange(n)
    for branch in range(2):
        for spin in range(2):
            col=2*branch*n+2*q+spin
            u[2*spin*n+q,col]=1/np.sqrt(2);u[(2*spin+1)*n+q,col]=(1 if branch==0 else -1)/np.sqrt(2)
    return u


def stability(p,B):
    a=(p[0]+p[1])/2
    slope=float('inf') if p[7]==0 else float(min(p[:2])/a*np.sqrt(a/p[7]))
    bound=float(6*abs(p[8])*max(np.hypot(p[3],p[4]),np.hypot(p[5],p[6]))*np.linalg.norm(B[0]))
    return dict(minimum_kinetic_slope_eV_A=slope if np.isfinite(slope) else None,SOC_slope_upper_bound_eV_A=bound,positive_margin=bool(slope>bound),margin_eV_A=float(slope-bound) if np.isfinite(slope) else None)
