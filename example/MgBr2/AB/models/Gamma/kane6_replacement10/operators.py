"""Six fields: j1/2 plus both j3/2 doublets; eight explicit homogeneous terms."""
import numpy as np
from scipy.linalg import block_diag
I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]],complex);Z=np.diag([1.,-1.]).astype(complex)
J=block_diag(*([1j*Y]*3));D=np.diag([np.exp(1j*np.pi/3),np.exp(-1j*np.pi/3),-1,-1,-1,-1])
HOM_NAMES=['A_half_eV_A2','A_HH_shared_eV_A2','Delta_half_eV','Delta_lower_HH_eV','v_lower_eV_A','v_upper_eV_A','b_lower_eV_A2','b_upper_eV_A2']
EVEN=[];POT_BASE_NAMES=[]
for a in range(3):
    M=np.zeros((6,6),complex);M[2*a:2*a+2,2*a:2*a+2]=I;EVEN.append(M);POT_BASE_NAMES.append(f'diag_orbital{a}')
for a,b in [(0,1),(0,2),(1,2)]:
    for label,T in [('real_I',I),('imag_sigma_x',1j*X),('imag_sigma_y',1j*Y),('imag_sigma_z',1j*Z)]:
        M=np.zeros((6,6),complex);M[2*a:2*a+2,2*b:2*b+2]=T;M[2*b:2*b+2,2*a:2*a+2]=T.conj().T;EVEN.append(M);POT_BASE_NAMES.append(f'cross_{a}{b}_{label}')
POT_NAMES=[f'{f}_{label}'for f in ['cos','sin']for label in POT_BASE_NAMES]

def homogeneous(p,kind='b_lower'):
    x,y=p;kp=x+1j*y;km=kp.conjugate();q2=x*x+y*y;O=np.zeros((8,6,6),complex)
    O[0,:2,:2]=q2*I;O[1,2:,2:]=q2*np.eye(4);O[2,:2,:2]=I;O[3,2:4,2:4]=I
    O[4,:2,2:4]=np.diag([km,-kp]);O[5,:2,4:6]=np.diag([km,-kp])
    O[6,:2,2:4]=np.array([[0,kp**2],[-km**2,0]]) if kind=='b_lower' else np.diag([-1j*kp**2,1j*km**2])
    O[7,:2,4:6]=np.array([[0,kp**2],[-km**2,0]])
    O[:,2:,:2]=O[:,:2,2:].conj().transpose(0,2,1);return O

def potential_operators(q,B):
    q=np.asarray(q);n=len(q);O=np.zeros((30,6*n,6*n),complex);diff=q[:,None]-q[None,:];star=np.array([B[0],B[1]-B[0],-B[1]])
    for j,g in enumerate(star):
        Dj=np.linalg.matrix_power(D,j);plus=(np.linalg.norm(diff-g,axis=-1)<1e-7).astype(float);c=plus+plus.T;s=-1j*(plus-plus.T)
        for a,M in enumerate(EVEN):
            R=Dj@M@Dj.conj().T;O[a]+=np.kron(c,R);O[a+15]+=np.kron(s,R)
    return O

def all_operators(q,B,k,kind='b_lower'):
    q=np.asarray(q);n=len(q);O=np.zeros((38,6*n,6*n),complex)
    for j,p in enumerate(q+np.asarray(k)@B):O[:8,6*j:6*j+6,6*j:6*j+6]=homogeneous(p,kind)
    O[8:]=potential_operators(q,B);return O

def physical(z,scales):
    p=np.asarray(z)*scales;K=np.diag(scales);az=5*z[1];rad=np.sqrt(az*az+p[6]**2+p[7]**2+1e-10);dm=5*np.exp(z[0]);a0=rad+.01+dm
    p[0]=a0+az;p[1]=a0-az;K[:2,:]=0;K[:2,0]=dm;K[0,1]=5*(az/rad+1);K[1,1]=5*(az/rad-1)
    K[:2,6]=p[6]*scales[6]/rad;K[:2,7]=p[7]*scales[7]/rad
    dl=.02*np.exp(z[3]);dh=.03*np.exp(z[2]);p[3]=.001+dl;p[2]=p[3]+.005+dh;K[2:4,:]=0;K[3,3]=dl;K[2,2]=dh;K[2,3]=dl
    return p,K

def coordinates(p,scales):
    z=np.asarray(p)/scales;az=(p[0]-p[1])/2;a0=(p[0]+p[1])/2;rad=np.sqrt(az**2+p[6]**2+p[7]**2+1e-10)
    z[0]=np.log(max(a0-rad-.01,1e-4)/5);z[1]=az/5;z[2]=np.log(max(p[2]-p[3]-.005,1e-4)/.03);z[3]=np.log(max(p[3]-.001,1e-4)/.02);return z

def principal(p):
    return float(min(p[1],(p[0]+p[1])/2-np.sqrt(((p[0]-p[1])/2)**2+p[6]**2+p[7]**2)))
