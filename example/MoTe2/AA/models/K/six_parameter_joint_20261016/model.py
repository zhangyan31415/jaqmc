"""Four-parameter MoTe2 K continuum family, with electron/hole evaluators.

Coefficients: a_h (eV A^2), V*cos(phi), V*sin(phi), w_h (eV).
The source electron envelope has exp(i*(k-Q_e).r). In the periodic layer gauge
Q_e=kappa-G, electron tunneling is -w_h*(1+exp(-i*g2.r)+exp(-i*g3.r)).
The hole basis is the conjugate electron basis, Q_h=-Q_e, k_h=-k_e.
"""
from pathlib import Path
import numpy as np

HBAR2=3.80998212
NAMES=('a_h','V_cos_phi','V_sin_phi','w_h')


def response(kpoints,q1,q2,reciprocal,kappas=None):
    k=np.atleast_2d(kpoints); q1=np.asarray(q1);q2=np.asarray(q2);b=np.asarray(reciprocal)
    n1=len(q1);n2=len(q2);n=n1+n2;q=np.vstack([q1,q2])
    kap=np.array([q1[0],q2[0]]) if kappas is None else np.asarray(kappas)
    out=np.zeros((len(k),4,n,n),complex)
    out[:,0,np.arange(n),np.arange(n)]=-np.sum((k[:,None]-q[None])**2,axis=-1)
    star=np.array([b[0],b[1]-b[0],-b[1]])
    for layer,qq in enumerate((q1,q2)):
        d=qq[:,None]-qq[None,:]
        plus=np.sum(np.linalg.norm(d[:,:,None]-star,axis=-1)<1e-8,axis=-1)
        minus=np.sum(np.linalg.norm(d[:,:,None]+star,axis=-1)<1e-8,axis=-1)
        sl=slice(0,n1) if layer==0 else slice(n1,n)
        out[:,1,sl,sl]=plus+minus
        out[:,2,sl,sl]=(-1j if layer==0 else 1j)*(plus-minus)
    delta=kap[0]-kap[1]
    transfers=np.array([delta,delta+b[1],delta+b[1]-b[0]])
    tunnel=np.sum(np.linalg.norm((q1[:,None]-q2[None])[:,:,None]-transfers,axis=-1)<1e-8,axis=-1)
    out[:,3,:n1,n1:]=-tunnel;out[:,3,n1:,:n1]=-tunnel.T
    return out


def extended_response(kpoints,q1,q2,reciprocal,kappas,family):
    k=np.atleast_2d(kpoints);q1=np.asarray(q1);q2=np.asarray(q2);b=np.asarray(reciprocal);kap=np.asarray(kappas)
    first=response(k,q1,q2,b,kap);n1=len(q1);n2=len(q2);n=n1+n2
    out=np.zeros((len(k),6,n,n),complex);out[:,:4]=first
    if family=='second_harmonic':
        star=np.array([b[0]+b[1],b[1]-2*b[0],b[0]-2*b[1]])
        for layer,q in enumerate((q1,q2)):
            d=q[:,None]-q[None,:]
            mask=np.sum(np.linalg.norm(d[:,:,None]-star,axis=-1)<1e-8,axis=-1)+np.sum(np.linalg.norm(d[:,:,None]+star,axis=-1)<1e-8,axis=-1)
            sl=slice(0,n1) if layer==0 else slice(n1,n)
            out[:,4,sl,sl]=mask
        delta=kap[0]-kap[1];nearest=np.array([delta,delta+b[1],delta+b[1]-b[0]])
        mask=np.sum(np.linalg.norm((q1[:,None]-q2[None])[:,:,None]+2*nearest,axis=-1)<1e-8,axis=-1)
        out[:,5,:n1,n1:]=-mask;out[:,5,n1:,:n1]=-mask.T
    elif family=='momentum_tunneling':
        d=q1[:,None]-q2[None,:];p=k[:,None,None]-(q1[:,None]+q2[None,:])[None]/2
        cross=(p[...,0]*d[None,...,1]-p[...,1]*d[None,...,0])*(-first[0,3,:n1,n1:])[None]
        out[:,4,:n1,n1:]=cross;out[:,4,n1:,:n1]=cross.transpose(0,2,1)
        out[:,5,:n1,n1:]=1j*cross;out[:,5,n1:,:n1]=-1j*cross.transpose(0,2,1)
    else:raise ValueError('Unknown six-parameter family')
    return out


class SimplifiedModel:
    def __init__(self,directory):
        with np.load(Path(directory)/'model.npz') as z:
            self.parameters=z['parameters'].copy();self.q1=z['q1'].copy();self.q2=z['q2'].copy()
            self.reciprocal=z['reciprocal'].copy();self.kappas=z['kappas'].copy()
            self.fractional_to_model=z['fractional_to_model'].copy()
            self.family=str(z['family'].item()) if 'family' in z else 'four_parameter'

    def electron_hamiltonian(self,k):
        b=response(np.asarray(k)[None],self.q1,self.q2,self.reciprocal,self.kappas) if self.family=='four_parameter' else extended_response(np.asarray(k)[None],self.q1,self.q2,self.reciprocal,self.kappas,self.family)
        return np.einsum('pij,p->ij',b[0],self.parameters)

    def hole_hamiltonian(self,k):
        top=np.linalg.eigvalsh(self.electron_hamiltonian([0.,0.]))[-1]
        return top*np.eye(len(self.q1)+len(self.q2))-self.electron_hamiltonian(-np.asarray(k)).T

    def hamiltonian(self,k,carrier='electron',coordinates='model'):
        k=np.asarray(k)
        if coordinates=='fractional':k=k@self.fractional_to_model
        elif coordinates!='model':raise ValueError('coordinates must be model or fractional')
        if carrier=='electron':return self.electron_hamiltonian(k)
        if carrier=='hole':return self.hole_hamiltonian(k)
        raise ValueError('carrier must be electron or hole')

    def bands(self,kpoints,carrier='electron',coordinates='model'):
        return np.array([np.linalg.eigvalsh(self.hamiltonian(k,carrier,coordinates)) for k in kpoints])

    def physical_parameters(self):
        a,vc,vs,w=self.parameters[:4]
        result=dict(m_h_over_me=float(HBAR2/a),V_mev=float(np.hypot(vc,vs)*1000),phi_deg=float(np.degrees(np.arctan2(vs,vc))),w_h_mev=float(w*1000))
        if self.family=='second_harmonic':result.update(V2_mev=float(self.parameters[4]*1000),w2_h_mev=float(self.parameters[5]*1000))
        if self.family=='momentum_tunneling':result.update(beta_e_real_ev_angstrom2=float(self.parameters[4]),beta_e_imag_ev_angstrom2=float(self.parameters[5]))
        return result
