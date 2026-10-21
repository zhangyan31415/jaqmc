"""Full continuum Bloch Slater initializer, not a production band projection."""
import numpy as np
from nqs18 import Model,N,jax,jnp,get_reciprocal_vectors


def build(shell=5):
    m=Model(walkers=1,covariant=True)
    b=np.asarray(get_reciprocal_vectors(m.primitive))
    klist=np.asarray(m.wf.multiphase_klist)
    labels=np.array([(i,j) for i in range(-shell,shell+1) for j in range(-shell,shell+1)
                     if max(abs(i),abs(j),abs(i+j))<=shell])
    g=labels@b; ng=len(g); index={tuple(z):i for i,z in enumerate(labels)}
    eig=[]; coeff=[]
    kp=(b[0]+b[1])/3; km=(2*b[0]-b[1])/3
    for k in klist:
        h=np.diag(np.concatenate([.5*m.pref*np.sum((k+g-kp)**2,axis=1),
                                  .5*m.pref*np.sum((k+g-km)**2,axis=1)])).astype(complex)
        for a,z in enumerate(labels):
            for q in [(1,0),(-1,1),(0,-1)]:
                dest=index.get(tuple(z+q))
                if dest is not None:
                    for l,phase in [(0,m.cfg.phi1_rad),(1,-m.cfg.phi1_rad)]:
                        h[l*ng+dest,l*ng+a]+=-m.cfg.v1_mev*np.exp(1j*phase)
                        h[l*ng+a,l*ng+dest]+=-m.cfg.v1_mev*np.exp(-1j*phase)
            for q in [(0,0),(0,1),(-1,1)]:
                dest=index.get(tuple(z+q))
                if dest is not None:
                    h[dest,ng+a]+=m.cfg.omega1_mev
                    h[ng+a,dest]+=m.cfg.omega1_mev
        e,c=np.linalg.eigh(h); eig.append(e[:3]); coeff.append(c[:,0].reshape(2,ng))
    eig=np.asarray(eig)
    # The finite reciprocal group has27 elements represented modulo27.
    tags=np.rint(klist@np.linalg.inv(b)*27).astype(int)%27
    table={(0,0,0):(0.,())}
    for i,(tag,energy) in enumerate(zip(tags,eig[:,0])):
        updated=dict(table)
        for (n,a,c),(cost,occ) in table.items():
            if n>=N:continue
            key=(n+1,(a+int(tag[0]))%27,(c+int(tag[1]))%27)
            val=(cost+float(energy),occ+(i,))
            if key not in updated or val[0]<updated[key][0]:updated[key]=val
        table=updated
    cost,occupied=table[(N,0,0)]
    occupied=np.asarray(occupied)
    c=np.stack([coeff[i] for i in occupied],axis=-1)
    return dict(g=g,coeff=c,occupied=occupied,klist=klist,eigenvalues=eig,
                total_band_energy=np.array(cost),shell=np.array(shell),tags=tags)


def initialize(model,seed,data):
    if model.wf.ndets != 1:
        raise ValueError('This Bloch initializer supports one base determinant.')
    p,r,s=model.initialize(seed)
    # A small neural residual around the Bloch Slater-Jastrow determinant.
    def scale(path,x):
        name='/'.join(str(getattr(z,'key',z)) for z in path)
        return x*1e-3 if 'orbital_layer_' in name else x
    p=jax.tree_util.tree_map_with_path(scale,p)
    from flax.core import unfreeze,freeze
    mutable=unfreeze(p)
    matrix=np.zeros((27,N))
    matrix[np.asarray(data['occupied']),np.arange(N)]=1
    mutable['params']['multiphase_layer']['multik']=jnp.asarray(matrix)
    return freeze(mutable),r,s

if __name__=='__main__':
    from pathlib import Path
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    data=build();np.savez_compressed(args.out,**data)
    print('Occupied',data['occupied'],'sum energy',data['total_band_energy'],flush=True)
