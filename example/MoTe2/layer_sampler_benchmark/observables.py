"""Physical two-layer observables for a sum of projected determinants."""
from nqs18 import N,coefficients,log_contract,jax,jnp,np


def make_observables(ex):
    q=2*jnp.pi*ex.invcell.T
    qs=jnp.stack([jnp.zeros(2),q[0],q[1],q[0]-q[1]])
    def one(phi,chi,r):
        mat=jnp.einsum('il,ldij->dij',chi,phi)
        zmat=jnp.einsum('il,l,ldij->dij',chi,jnp.array([1.,-1.]),phi)
        signs,logs=jnp.linalg.slogdet(mat);offset=jnp.max(logs)
        weights=signs*jnp.exp(logs-offset);weights=weights/weights.sum()
        a=zmat@jnp.linalg.inv(mat)
        diagonals=jnp.diagonal(a,axis1=-2,axis2=-1)
        z=jnp.einsum('d,di->i',weights,diagonals)
        pair=diagonals[:,:,None]*diagonals[:,None,:]-a*jnp.swapaxes(a,-1,-2)
        zz=jnp.einsum('d,dij->ij',weights,pair).at[jnp.diag_indices(N)].set(1.)
        phase=jnp.exp(1j*(r@qs.T))
        spin_modes=jnp.mean(z[:,None]*phase,axis=0)
        spin_s=jnp.einsum('iq,ij,jq->q',phase.conj(),zz,phase).real/N
        charge_modes=jnp.mean(phase,axis=0);charge_s=N*abs(charge_modes)**2
        return jnp.concatenate([jnp.array([jnp.mean(z).real,jnp.mean(zz).real]),
            spin_s,spin_modes.real,spin_modes.imag,charge_s,charge_modes.real,charge_modes.imag])
    return jax.jit(jax.vmap(one)),one


def validate_observable(ex,p,observable):
    r,u,phi,key=ex.initial(p,9191,True);r,u,phi=r[0],u[0],phi[0]
    chi=coefficients(u);mat=jnp.einsum('il,ldij->dij',chi,phi)
    zmat=jnp.einsum('il,l,ldij->dij',chi,jnp.array([1.,-1.]),phi)
    def logsum(matrix):
        signs,logs=jnp.linalg.slogdet(matrix);offset=jnp.max(logs)
        return offset+jnp.log(jnp.sum(signs*jnp.exp(logs-offset)))
    base=logsum(mat);exact=jnp.eye(N,dtype=jnp.complex128)
    for i in range(N):
        for j in range(i):
            swapped=mat.at[:,i,:].set(zmat[:,i,:]).at[:,j,:].set(zmat[:,j,:])
            exact=exact.at[i,j].set(jnp.exp(logsum(swapped)-base))
            exact=exact.at[j,i].set(exact[i,j])
    actual=observable(phi,chi,r)
    np.testing.assert_allclose(actual[1],exact.real.mean(),rtol=2e-8,atol=2e-8)
    return float(abs(actual[1]-exact.real.mean()))
