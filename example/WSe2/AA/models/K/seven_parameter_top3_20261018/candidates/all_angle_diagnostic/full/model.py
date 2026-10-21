"""Pure quadratic electron kinetic energy and one real linear tunneling amplitude."""
from pathlib import Path
import numpy as np

HBAR2=3.80998212
NAMES=['a_h','Vcos','Vsin','w1','V2','w2','lambda_eV_A']

def response7(kpoints,q1,q2,b,kappas,theta_deg):
    k=np.atleast_2d(kpoints);q1=np.asarray(q1);q2=np.asarray(q2);b=np.asarray(b)
    n1=len(q1);n2=len(q2);n=n1+n2;q=np.vstack([q1,q2])
    out=np.zeros((len(k),7,n,n),complex)
    out[:,0,np.arange(n),np.arange(n)]=-np.sum((k[:,None]-q[None])**2,axis=-1)
    for layer,qq in enumerate([q1,q2]):
        delta=qq[:,None]-qq[None,:];sl=slice(0,n1) if layer==0 else slice(n1,n)
        star=np.array([b[0],b[1]-b[0],-b[1]])
        plus=np.sum(np.linalg.norm(delta[:,:,None]-star,axis=-1)<1e-8,axis=-1)
        minus=np.sum(np.linalg.norm(delta[:,:,None]+star,axis=-1)<1e-8,axis=-1)
        out[:,1,sl,sl]=plus+minus
        out[:,2,sl,sl]=(-1j if layer==0 else 1j)*(plus-minus)
        star2=np.array([b[0]+b[1],b[1]-2*b[0],b[0]-2*b[1]])
        out[:,4,sl,sl]=np.sum(np.linalg.norm(delta[:,:,None]-star2,axis=-1)<1e-8,axis=-1)+np.sum(np.linalg.norm(delta[:,:,None]+star2,axis=-1)<1e-8,axis=-1)
    d=np.asarray(kappas)[0]-np.asarray(kappas)[1]
    transfers=np.array([d,d+b[1],d+b[1]-b[0]])
    delta=q1[:,None]-q2[None,:]
    masks=np.linalg.norm(delta[:,:,None]-transfers,axis=-1)<1e-8
    out[:,3,:n1,n1:]=-masks.sum(axis=-1);out[:,3,n1:,:n1]=-masks.sum(axis=-1).T
    second=np.sum(np.linalg.norm(delta[:,:,None]+2*transfers,axis=-1)<1e-8,axis=-1)
    out[:,5,:n1,n1:]=-second;out[:,5,n1:,:n1]=-second.T
    row=k[:,None]-q1[None];col=k[:,None]-q2[None]
    rp=row[...,0]+1j*row[...,1];cm=col[...,0]-1j*col[...,1]
    omega=np.exp(2j*np.pi/3);phase=np.exp(1j*np.deg2rad(theta_deg)/2)
    for j in range(3):
        c=omega**(-j)
        out[:,6,:n1,n1:]-=phase*masks[None,:,:,j]*(c*rp[:,:,None]+c.conjugate()*cm[:,None,:])
    out[:,6,n1:,:n1]=out[:,6,:n1,n1:].conj().transpose(0,2,1)
    return out

class Model:
    def __init__(self,directory):
        with np.load(Path(directory)/'model.npz') as z:
            for name in ['q1','q2','reciprocal','kappas','fractional_to_model','parameters','theta_deg']:
                setattr(self,name,z[name].copy())
    def hamiltonian(self,k,valley='K'):
        k=np.asarray(k)
        if valley not in ['K','Kprime']:raise ValueError(valley)
        h=np.einsum('pij,p->ij',response7(k if valley=='K' else -k,self.q1,self.q2,self.reciprocal,self.kappas,float(self.theta_deg))[0],self.parameters)
        return h if valley=='K' else h.conj()
