"""Source-guided two-layer Gamma continuum, full Kramers sectors.
Units: eV, Angstrom. Real layer-exchange potentials are common to both layers.
Parity blocks are an exact computational decomposition of the stated model.
"""
import numpy as np
PHYSICAL_NAMES=['a_eV_A2','b_eV_inverse','eta_kinetic_tunneling','lambda_A2','Vc_eV','Vs_eV','t0_eV','tc_eV','ts_eV','epsilon_mass_anisotropy']
HBAR2=3.80998212
RMIN=(np.sin(np.deg2rad(2.28)/2)/np.sin(np.deg2rad(3)/2))**2
RMAX=(np.sin(np.deg2rad(9.43)/2)/np.sin(np.deg2rad(3)/2))**2

def parameters(c,theta):
 c=np.asarray(c);n=9 if len(c)==14 else 10;start=4 if n==9 else 5;p=np.zeros(n);j=np.zeros((n,len(c)));p[:4]=c[:4];j[:4,:4]=np.eye(4)
 if n==10:p[9]=c[4];j[9,4]=1
 r=(np.sin(np.deg2rad(theta)/2)/np.sin(np.deg2rad(3)/2))**2;u=(r-RMIN)/(RMAX-RMIN)
 for i in range(4,9):
  k=start+2*(i-4);p[i]=(1-u)*c[k]+u*c[k+1];j[i,k:k+2]=[1-u,u]
 return p,j

def kane(p2,a,b):
 root=np.sqrt(1+4*b*a*p2);e=2*a*p2/(1+root);return e,p2/root,-e*e/root

def geometry_operators(q,b,k):
 q=np.asarray(q);b=np.asarray(b);kc=np.asarray(k)@b;delta=q[:,None]-q[None,:];pavg=(q[:,None]+q[None,:])/2+kc;cross=delta[:,:,0]*pavg[:,:,1]-delta[:,:,1]*pavg[:,:,0];stars=[np.array([b[0],b[1]-b[0],-b[1]]),np.array([b[0]+b[1],b[1]-2*b[0],b[0]-2*b[1]])];o=[]
 for star in stars:
  plus=sum(np.max(abs(delta-v),axis=2)<1e-7 for v in star);minus=sum(np.max(abs(delta+v),axis=2)<1e-7 for v in star);o.extend([(plus+minus).astype(complex),1j*(plus-minus)])
 c,s,c2,s2=o;momentum=q+kc;p2=np.sum(momentum**2,axis=1);axis=(b[1]-b[0])/np.linalg.norm(b[1]-b[0]);quadrupole=2*(momentum@axis)**2-p2;return p2,np.array([c,s,c2,1j*c*cross,1j*s*cross,1j*c2*cross]),quadrupole

def from_geometry(geom,p,sector,parity):
 p2,o,quadrupole=geom;a,b,eta,lam,vc,vs,t0,tc,ts=p[:9];eps=p[9] if len(p)>9 else 0.;argument=p2+eps*quadrupole;e,da,db=kane(argument,a,b);wc=vc+parity*tc;ws=vs+parity*ts;pot=wc*o[0]+ws*o[1];soc=wc*o[3]+ws*o[4];n=len(p2);ident=np.eye(n);h=np.diag((1+parity*eta)*e)+parity*t0*ident+pot+sector*lam*soc
 d=np.array([np.diag((1+parity*eta)*da),np.diag((1+parity*eta)*db),np.diag(parity*e),sector*soc,o[0]+sector*lam*o[3],o[1]+sector*lam*o[4],parity*ident,parity*(o[0]+sector*lam*o[3]),parity*(o[1]+sector*lam*o[4])]+([np.diag((1+parity*eta)*a*quadrupole/np.sqrt(1+4*b*a*argument))] if len(p)>9 else []))
 return h,d

def block(q,b,k,p,sector,parity):return from_geometry(geometry_operators(q,b,k),p,sector,parity)

def full(q,b,k,p):
 n=len(q);h=np.zeros((4*n,4*n),complex)
 for sector,offset in [(1,0),(-1,2*n)]:
  hp,_=block(q,b,k,p,sector,1);hm,_=block(q,b,k,p,sector,-1);diag=(hp+hm)/2;off=(hp-hm)/2;h[offset:offset+n,offset:offset+n]=diag;h[offset+n:offset+2*n,offset+n:offset+2*n]=diag;h[offset:offset+n,offset+n:offset+2*n]=off;h[offset+n:offset+2*n,offset:offset+n]=off
 return h

def integer_shell(radius):
 return np.array([(i,j) for i in range(-radius,radius+1) for j in range(-radius,radius+1) if max(abs(i),abs(j),abs(i+j))<=radius],float)

def stability(p,b):
 # Matrix potential V I+T tau_x: maximize over its two parity eigenvalues.
 vc,vs,tc,ts=p[4],p[5],p[7],p[8];g=np.linalg.norm(b[0]);bound=abs(p[3])*max(6*abs(complex(vc+tau*tc,vs+tau*ts))*g for tau in [-1,1]);slope=float('inf') if p[1]==0 else (1-abs(p[2]))*np.sqrt(p[0]*(1-abs(p[9]) if len(p)>9 else 1)/p[1]);return dict(minimum_Kane_asymptotic_slope_eV_A=float(slope),SOC_linear_norm_upper_bound_eV_A=float(bound),strict_positive_margin_eV_A=float(slope-bound))
