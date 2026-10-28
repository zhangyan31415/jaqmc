"""Spin-free local M-valley kernels. Energies eV, lengths Angstrom."""
import numpy as np

NAMES=['A_parallel_eV_A2','A_perp_eV_A2','w_a_eV','w_b_eV','Sd_eV','Va_eV','Vd_eV','Sa_eV','Sl_eV']
SELECT={9:np.arange(9),7:np.array([0,1,2,3,4,5,7])}

def native_density(c,left,right):
    cs=c[left:right].reshape(right-left,-1,2)
    return np.einsum('aqs,ars->qr',cs.conj(),cs)/2

def native_scales(native):
    centers=np.asarray(native)[...,:6].reshape(-1,3,2).mean(axis=-1)
    return float(np.ptp(centers)),np.sqrt(np.mean(np.diff(centers,axis=1)**2,axis=0))

def probability_gradient(e,u,a,left,right,ops):
    rank=right-left;ui=u[:,left:right];p=float(np.trace(ui.conj().T@a@ui).real/rank)
    outside=np.r_[0:left,right:len(e)];uo=u[:,outside]
    den=e[left:right][None,:]-e[outside][:,None]
    if np.min(abs(den))<1e-11:raise FloatingPointError('Unresolved orbital model group boundary')
    coeff=(uo.conj().T@a@ui)/den
    gh=uo@coeff@ui.conj().T
    return p,2*np.einsum('ij,aij->a',gh.conj(),ops,optimize=True).real/rank

def operators(g,k,nparam,geometry=None):
    # Accept precomputed source geometry to avoid changing Fourier conventions.
    if geometry is None:
        import physical
        geometry=physical.m_geometry(g,k)
    raw=geometry[0]
    ops=raw[[0,1,6,7,2,4,5,8,9],::2,::2].copy()
    if nparam==7:
        kap=np.asarray(g['kappa1'])+np.asarray(g['kappa2']);e=kap/np.linalg.norm(kap);ep=np.array([-e[1],e[0]])
        p=np.asarray(k)@np.asarray(g['B'])-np.concatenate([g['q1'],g['q2']])
        ops[0]=np.diag((p@e)**2);ops[1]=np.diag((p@ep)**2)
    return ops[SELECT[nparam]]

def potentials(r,B,kappa1,kappa2,p,nparam,stars):
    """Layer potentials and real interlayer tunneling in the source envelope gauge."""
    full=np.zeros(9);full[SELECT[nparam]]=p
    _,_,wa,wb,sd,va,vd,sa,sl=full
    axis,obs=stars(np.asarray(B),np.asarray(kappa1)-kappa2,np.asarray(kappa1)+kappa2)
    r=np.asarray(r);aa=r@obs[0];bb=r@obs[1];cc=r@axis
    even=2*sd*(np.sin(aa)-np.sin(bb))+2*va*np.cos(cc)+2*sa*np.sin(cc)
    odd=2*vd*(np.cos(aa)-np.cos(bb))+2*sl*(np.sin(aa)+np.sin(bb))
    qa=(B[0]-B[1])/2;qb=(B[0]+B[1])/2
    tunnel=2*wa*np.cos(r@qa)+2*wb*np.cos(r@qb)
    return np.stack([even+odd,even-odd,tunnel],axis=-1)
