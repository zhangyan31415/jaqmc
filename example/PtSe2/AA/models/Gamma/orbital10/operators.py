import numpy as np
SX=np.array([[0,1],[1,0]],complex);SY=np.array([[0,-1j],[1j,0]],complex);SZ=np.diag([1.,-1.]);I=np.eye(2)
def base_operators(q,B,k,_=None):
 q=np.asarray(q);n=len(q);dim=4*n;p=q+np.asarray(k)@B;ops=np.zeros((10,dim,dim),complex);SOC=np.kron(SY,SZ)
 def onsite(index,qi,m):
  rr=[2*qi,2*qi+1,2*(n+qi),2*(n+qi)+1];ops[index][np.ix_(rr,rr)]+=m
 def harmonic(index,mask,m):
  for a in range(2):
   for b in range(2):ops[index,2*a*n:2*(a+1)*n,2*b*n:2*(b+1)*n]+=np.kron(mask,m[2*a:2*a+2,2*b:2*b+2])
 for qi,(px,py)in enumerate(p):
  quad=np.kron((px*px-py*py)*SZ+2*px*py*SX,I);base=(px*px+py*py)*np.eye(4);onsite(0,qi,(base+quad)/2);onsite(1,qi,(base-quad)/2);onsite(2,qi,SOC);onsite(9,qi,np.kron(I,px*SX+py*SY))
 diff=q[:,None]-q[None,:];star=np.array([B[0],B[1]-B[0],-B[1]])
 for g in star:
  plus=(np.max(abs(diff-g),axis=2)<1e-7).astype(float);c=plus+plus.T;s=-1j*(plus-plus.T);phi=np.arctan2(g[1],g[0]);M=np.kron(np.cos(2*phi)*SZ+np.sin(2*phi)*SX,I);harmonic(3,c,np.eye(4));harmonic(4,s,np.eye(4));harmonic(5,c,M);harmonic(6,s,M);harmonic(7,c,SOC);harmonic(8,s,SOC)
 return ops
def operators(q,B,k,_=None):
 o=base_operators(q,B,k);n=len(q);p=np.asarray(q)+np.asarray(k)@B;SOC=np.kron(SY,SZ);o[:2]=0
 for i,pp in enumerate(p):
  rows=[2*i,2*i+1,2*(n+i),2*(n+i)+1];p2=pp@pp;o[0][np.ix_(rows,rows)]=p2*(np.eye(4)+SOC)/2;o[1][np.ix_(rows,rows)]=p2*(np.eye(4)-SOC)/2
 return o

chiral_operators=operators
def operators(q, B, fractional, _=None):
    ops = chiral_operators(q, B, fractional)
    ops[9] = 0
    n = len(q)
    for i, (px, py) in enumerate(np.asarray(q) + np.asarray(fractional) @ B):
        rows = [2*i, 2*i+1, 2*(n+i), 2*(n+i)+1]
        ops[9][np.ix_(rows, rows)] = np.kron(px*SZ-py*SX, SZ)
    return ops
