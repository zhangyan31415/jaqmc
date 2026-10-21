"""Six independent real coefficients of the strict AB simplified model.

Basis: Q -> layer -> orbital -> spin. Cartesian momentum in inverse Angstrom,
Hamiltonian in eV. Both AB phases are identically zero. No energy constant.
"""
from pathlib import Path
import numpy as np
from scipy.linalg import eigh

NAMES = ('a1', 'a2', 'V1', 'V2', 'w1', 'w2')
HBAR2 = 3.80998212
LITERATURE = np.array([HBAR2/-.2376, HBAR2/-.7387, .04323, -.03566, .01208, -.01584])
PARAMETER_SCALE = np.array([20., 10., .05, .05, .025, .025])
I2 = np.eye(2); X = np.array([[0,1],[1,0]]); Y = np.array([[0,-1j],[1j,0]]); Z = np.diag([1.,-1.])


def rotation(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s],[s,c]])


def hex_q(basis, radius):
    return np.array([np.array([i,j])@basis for i in range(-radius,radius+1)
                     for j in range(-radius,radius+1) if max(abs(i),abs(j),abs(i+j))<=radius])


def response(kpoints, qpoints, first_vector):
    kpoints=np.asarray(kpoints);qpoints=np.asarray(qpoints); n=len(qpoints); dim=8*n
    out=np.zeros((len(kpoints),6,dim,dim),complex)
    g00=np.kron(I2,I2);gx=np.kron(X,I2);gyz=np.kron(Y,Z);gz=np.kron(Z,I2)
    star=np.array([rotation(j*2*np.pi/3)@first_vector for j in range(3)])
    matrices=[gyz, np.sqrt(3)/2*gx-.5*gyz, -np.sqrt(3)/2*gx-.5*gyz]
    for i,qi in enumerate(qpoints):
        px,py=(kpoints-qi).T; sl=slice(8*i,8*i+8)
        out[:,0,sl,sl]=(px*px+py*py)[:,None,None]*np.eye(8)
        out[:,1,sl,sl]=(px*px-py*py)[:,None,None]*np.kron(I2,gx)+2*(px*py)[:,None,None]*np.kron(I2,gyz)
        out[:,2,sl,sl]=np.kron(I2,gz)
        out[:,4,sl,sl]=np.kron(X,g00)
        for j,qj in enumerate(qpoints):
            if i==j: continue
            target=slice(8*j,8*j+8)
            for g,mat in zip(star,matrices):
                if min(np.linalg.norm(qi-qj-g),np.linalg.norm(qi-qj+g))<1e-8:
                    out[:,3,sl,target]+=np.kron(Z,mat)
                    out[:,5,sl,target]+=np.kron(X,g00)
    return out


def analytic_symmetries():
    tr=np.kron(I2,np.kron(I2,-1j*Y))
    c2=np.kron(X,np.kron(I2,-1j*Y))
    c3=np.kron(I2,np.diag([-1.,-1.,np.exp(1j*np.pi/3),np.exp(-1j*np.pi/3)]))
    return [('TR',-np.eye(2),tr,True),('C3',rotation(2*np.pi/3),c3,False),('C2',np.diag([-1.,1.]),c2,False)]


def full_symmetry(qpoints, action, internal):
    n=len(qpoints);rho=np.zeros((8*n,8*n),complex)
    for i,q in enumerate(qpoints):
        j=int(np.argmin(np.linalg.norm(qpoints-(action@q),axis=1)))
        if np.linalg.norm(qpoints[j]-action@q)>1e-8: raise ValueError('Q set not symmetry closed')
        rho[8*j:8*j+8,8*i:8*i+8]=internal
    return rho


def independent_count(basis):
    b=np.asarray(basis).reshape(6,-1).T
    return int(np.linalg.matrix_rank(np.vstack((b.real,b.imag)),tol=1e-10))


def split_path(kpoints, anchors=(0,20,40)):
    unique=[]
    for i,k in enumerate(kpoints):
        if not any(np.linalg.norm(k-kpoints[j])<1e-10 for j in unique): unique.append(i)
    train=[i for i in unique if i%3!=1 or i in anchors]
    return train,[i for i in unique if i not in train]


def spin_up_indices(nq):
    return np.array([8*q+j for q in range(nq) for j in (0,2,4,6)])


def band_residual_jacobian(basis, parameters, target_eigenvalues, rows, bands=6, scale=.001):
    """Gamma-relative energy loss, including the derivative of model Gamma top.

    A spin-up block may represent the exactly spin-degenerate full Hamiltonian;
    when the reference dimension is twice its dimension, repeat both eigenvalues
    and derivatives. The reference's actual small pair splittings remain in loss.
    """
    b=np.asarray(basis);dim=b.shape[-1];factor=2 if target_eigenvalues.shape[-1]==2*dim else 1
    if bands%factor: raise ValueError('Spin-degenerate window must contain complete doublets')
    h=np.einsum('kpij,p->kij',b,parameters,optimize=True)
    values={};derivatives={}
    for index in dict.fromkeys([0,*rows]):
        e,u=eigh(h[index],subset_by_index=[dim-bands//factor,dim-1],check_finite=False,driver='evr')
        j=np.einsum('in,pij,jn->np',u.conj(),b[index],u,optimize=True).real
        values[index]=np.repeat(e,factor); derivatives[index]=np.repeat(j,factor,axis=0)
    top=values[0][-1];jtop=derivatives[0][-1];ref_top=target_eigenvalues[0,-1]
    res=np.concatenate([values[i]-top-(target_eigenvalues[i,-bands:]-ref_top) for i in rows])
    jac=np.concatenate([derivatives[i]-jtop for i in rows])
    norm=scale*np.sqrt(len(res))
    return res/norm,jac/norm


class SimplifiedModel:
    """Standalone adjustable six-parameter physical Hamiltonian."""
    def __init__(self, directory):
        with np.load(Path(directory)/'model.npz') as z:
            self.parameters=z['parameters'].copy(); self.q=z['q'].copy();self.first=z['first'].copy();self.coordinate_rotation=z['coordinate_rotation'].copy()
    def hamiltonian(self,k):
        k=np.asarray(k)@self.coordinate_rotation.T
        return np.einsum('pij,p->ij',response(k[None],self.q,self.first)[0],self.parameters)
    def bands(self,kpoints):
        return np.array([np.linalg.eigvalsh(self.hamiltonian(k)) for k in kpoints])


def reconstructed_overlap(native_map,gram,coefficients,pair_rows):
    """Statewise Feshbach cluster, Gram normalized after choosing the cluster."""
    metric=coefficients.conj().T@gram@coefficients
    e,u=np.linalg.eigh((metric+metric.conj().T)/2)
    if e[0]<=1e-12: raise ValueError('Singular reconstructed cluster')
    inverse=(u/np.sqrt(e)[None,:])@u.conj().T
    cross=native_map[pair_rows,:]@coefficients@inverse
    return float(np.sqrt(np.sum(abs(cross)**2)/coefficients.shape[1])),float(np.linalg.svd(cross,compute_uv=False)[-1]**2)
