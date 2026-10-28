"""Simple two-branch Gamma and local spin/layer M operators, eV and Angstrom."""
import numpy as np
HBAR2=3.80998211615486
SX=np.array([[0,1],[1,0]],complex);SY=np.array([[0,-1j],[1j,0]],complex);SZ=np.diag([1.,-1.])
GAMMA_NAMES=['A_plus_eV_A2','A_minus_eV_A2','Delta_eV','Vc_plus_eV','Vs_plus_eV','Vc_minus_eV','Vs_minus_eV','b_eV_inverse']
M_POOL_NAMES={2:'S_oblique_difference_eV',4:'V_axis_eV',5:'V_oblique_layer_difference_eV',8:'S_axis_eV',9:'S_oblique_layer_sum_eV'}


def gamma_geometry(q,B,k):
    q=np.asarray(q);p2=np.sum((q+np.asarray(k)@B)**2,axis=1);d=q[:,None]-q[None,:]
    star=np.array([B[0],B[1]-B[0],-B[1]]);plus=sum((np.max(abs(d-a),axis=2)<1e-7).astype(float) for a in star)
    return p2,plus+plus.T,-1j*(plus-plus.T)


def gamma_from_geometry(g,p):
    p2,c,s=g;n=len(p2);p=np.asarray(p);ap,am,delta,cp,sp,cm,sm=p[:7];mean=(ap+am)/2;b=p[7] if len(p)==8 else 0.
    root=np.sqrt(1+4*b*mean*p2);f=2*mean*p2/(1+root);df=p2/root;db=-f*f/root
    h=np.zeros((2*n,2*n),complex);j=np.zeros((len(p),2*n,2*n),complex)
    for branch,(a,vc,vs) in enumerate([(ap,cp,sp),(am,cm,sm)]):
        sl=slice(branch*n,(branch+1)*n);h[sl,sl]=np.diag(a/mean*f+(delta if branch else 0.))+vc*c+vs*s
        common=a*(df/mean-f/(mean*mean))/2
        j[0,sl,sl]=np.diag(common+(f/mean if branch==0 else 0.));j[1,sl,sl]=np.diag(common+(f/mean if branch==1 else 0.))
        if branch:j[2,sl,sl]=np.eye(n)
        j[3+2*branch,sl,sl]=c;j[4+2*branch,sl,sl]=s
        if len(p)==8:j[7,sl,sl]=np.diag(a/mean*db)
    return h,j


def gamma_hj(q,B,k,p):return gamma_from_geometry(gamma_geometry(q,B,k),p)


def gamma_potentials(r,B,p):
    r=np.asarray(r);star=np.array([B[0],B[1]-B[0],-B[1]]);a=r@star.T;c=2*np.cos(a).sum(axis=-1);s=2*np.sin(a).sum(axis=-1)
    return np.stack([p[3]*c+p[4]*s,p[2]+p[5]*c+p[6]*s],axis=-1)


def gamma_vectors(v):
    n=len(v)//2;out=np.zeros((4*n,2*v.shape[1]),complex)
    for spin in range(2):
        out[2*spin*n:(2*spin+1)*n,spin::2]=(v[:n]+v[n:])/np.sqrt(2)
        out[(2*spin+1)*n:(2*spin+2)*n,spin::2]=(v[:n]-v[n:])/np.sqrt(2)
    return out


def gamma_targets(c):
    n=c.shape[1]//4;cs=[np.concatenate([(c[:,2*s*n:(2*s+1)*n]+c[:,(2*s+1)*n:(2*s+2)*n])/np.sqrt(2),(c[:,2*s*n:(2*s+1)*n]-c[:,(2*s+1)*n:(2*s+2)*n])/np.sqrt(2)],axis=1) for s in range(2)]
    return [sum(x[:2*r].conj().T@x[:2*r] for x in cs)/2 for r in [1,2,3]]


def stars(B,delta,kappa_sum):
    vectors=np.array([i*B[0]+j*B[1] for i in range(-2,3) for j in range(-2,3) if i or j]);norm=np.linalg.norm(vectors,axis=1);short=vectors[abs(norm-norm.min())<1e-7]
    positive=np.array([v for v in short if tuple(np.round(v,10))>tuple(np.round(-v,10))]);i=int(np.argmax(abs(positive@delta)));axis=positive[i]
    if axis@delta<0:axis=-axis
    oblique=np.delete(positive,i,axis=0);pairs=[]
    for a in [oblique[0],-oblique[0]]:
        for b in [oblique[1],-oblique[1]]:
            if np.linalg.norm(a-b-axis)<1e-7:pairs.append([a,b])
            if np.linalg.norm(b-a-axis)<1e-7:pairs.append([b,a])
    pair=max(pairs,key=lambda ab:np.sum(ab,axis=0)@kappa_sum)
    return axis,np.array(pair)


def m_geometry(g,k):
    q=[np.asarray(g['q1']),np.asarray(g['q2'])];kap=[np.asarray(g['kappa1']),np.asarray(g['kappa2'])];B=np.asarray(g['B']);n=[len(a) for a in q];off=2*n[0];dim=2*sum(n);o=np.zeros((11,dim,dim),complex)
    axis,obs=stars(B,kap[0]-kap[1],kap[0]+kap[1]);eye=np.eye(2)
    parallel=np.zeros((dim,dim),complex)
    for layer in range(2):
        start=0 if layer==0 else off;sl=slice(start,start+2*n[layer]);e=kap[layer]/np.linalg.norm(kap[layer]);ep=np.array([-e[1],e[0]]);p=np.asarray(k)@B-q[layer];pp,pt=p@e,p@ep;eta=1 if layer==0 else -1
        o[0,sl,sl]=np.diag(np.repeat(pp**2,2));o[1,sl,sl]=np.diag(np.repeat(pt**2,2));o[3,sl,sl]=-eta*np.kron(np.diag(pt),e[0]*SX+e[1]*SY)
        parallel[sl,sl]=eta*np.kron(np.diag(pp),ep[0]*SX+ep[1]*SY)
        d=q[layer][None]-q[layer][:,None]
        def cosmask(v):return (np.max(abs(d-v),axis=2)<1e-7).astype(float)+(np.max(abs(d+v),axis=2)<1e-7).astype(float)
        def sinmask(v):return -1j*((np.max(abs(d-v),axis=2)<1e-7).astype(float)-(np.max(abs(d+v),axis=2)<1e-7).astype(float))
        o[4,sl,sl]=np.kron(cosmask(axis),eye);o[5,sl,sl]=eta*np.kron(cosmask(obs[0])-cosmask(obs[1]),eye)
        o[2,sl,sl]=np.kron(sinmask(obs[0])-sinmask(obs[1]),eye);o[8,sl,sl]=np.kron(sinmask(axis),eye);o[9,sl,sl]=eta*np.kron(sinmask(obs[0])+sinmask(obs[1]),eye)
    d=q[0][:,None]-q[1][None];r=np.linalg.norm(d,axis=2);levels=[]
    for a in np.sort(r.ravel()):
        if not levels or abs(a-levels[-1])>1e-7:levels.append(float(a))
    masks=[(abs(r-a)<1e-7).astype(float) for a in levels[:2]]
    for j,mask in enumerate(masks,6):o[j,:off,off:]=np.kron(mask,eye);o[j,off:,:off]=o[j,:off,off:].conj().T
    q0=(B[0]-B[1])/2
    if q0@(kap[0]-kap[1])<0:q0=-q0
    odd=1j*((np.max(abs(d-q0),axis=2)<1e-7).astype(float)-(np.max(abs(d+q0),axis=2)<1e-7).astype(float))
    o[10,:off,off:]=np.kron(odd,eye);o[10,off:,:off]=o[10,:off,off:].conj().T
    tse=np.zeros((dim,dim),complex);tse[:off,off:]=np.kron(masks[1],1j*SZ);tse[off:,:off]=tse[:off,off:].conj().T
    return o,parallel,tse


def default_m_config(key):
    return dict(material=key,potential_indices=[5,8,9],soc='velocity' if key=='sns2_m' else 'interlayer_sigma_z',odd_transfer=False)


def m_names(config):
    return ['A_parallel_eV_A2','A_perp_eV_A2','w_a_eV','w_b_eV']+[M_POOL_NAMES[i] for i in config['potential_indices']]+[('alpha_eV_A' if config['soc']=='velocity' else 'lambda_eV')]+(['w_odd_eV'] if config.get('odd_transfer') else [])


def m_from_geometry(g,p,config):
    o,parallel,tse=g;indices=[0,1,6,7]+config['potential_indices'];n=len(indices);j=np.array([o[i] for i in indices]+[o[3] if config['soc']=='velocity' else tse]+([o[10]] if config.get('odd_transfer') else []));h=np.einsum('a,aij->ij',p,j)
    if config['soc']=='velocity':
        h+=p[n]*p[0]/p[1]*parallel;j[0]+=p[n]/p[1]*parallel;j[1]-=p[n]*p[0]/p[1]**2*parallel;j[n]+=p[0]/p[1]*parallel
    return h,j


def m_hj(g,k,p,config):return m_from_geometry(m_geometry(g,k),p,config)
