"""SR and native complex KFAC using the same VMC force and trust limits."""
from helpers import jax,jnp,sr_direction,microbatch_map
from jax.flatten_util import ravel_pytree
from nqs18 import MoireData,coefficients


def clip_training_energy(e):
    med=jnp.median(e.real);mad=jnp.maximum(jnp.mean(abs(e.real-med)),.1)
    medi=jnp.median(e.imag);madi=jnp.maximum(jnp.mean(abs(e.imag-medi)),.1)
    return jnp.clip(e.real,med-5*mad,med+5*mad)+1j*jnp.clip(e.imag,medi-5*madi,medi+5*madi)


def vmc_force(logs_fn,params,energies):
    """Re <O* (E_local - mean E_local)>; half the real energy gradient."""
    centered=jax.lax.stop_gradient(energies-energies.mean())
    return jax.grad(lambda p:jnp.real(jnp.mean(jnp.conj(logs_fn(p))*centered)))(params)


class TrainingUpdate:
    def __init__(self,model,params,optimizer='sr',damping=.05,inverse_period=10,ema=.95):
        if optimizer not in ['sr','kfac']:raise ValueError(optimizer)
        self.model=model;self.name=optimizer
        self.damping=damping;self.inverse_period=inverse_period;self.ema=ema
        _,self.unravel=ravel_pytree(params)
        self.energy=model.batch_energy(8)
        if optimizer=='kfac':
            from jaqmc.optimizer.kfac import KFACOptimizer
            self.kfac=KFACOptimizer(f_log_psi=model.logdata,learning_rate=1.,
                pmap_axis_name=None,predictive_variance=1.,norm_constraint=None,
                reject_generic=True,
                damping=damping,curvature_ema=ema,inverse_update_period=inverse_period)
        self.step=jax.jit(self._step)
        self.force=jax.jit(self._force)

    def batch(self,r,u):
        from jaqmc.data import BatchedData
        return BatchedData(MoireData(positions=r,spin_coords=jax.vmap(coefficients)(u)),
                           ['positions','spin_coords'])

    def logs(self,p,r,chi):
        return jax.vmap(lambda a,b:self.model.logdata(p,MoireData(positions=a,spin_coords=b)))(r,chi)

    def init(self,p,r,u,key):
        return self.kfac.init(p,batched_data=self.batch(r,u),rngs=key) if self.name=='kfac' else None

    def _force(self,p,r,u,energies):
        chi=jax.vmap(coefficients)(u)
        return vmc_force(lambda q:self.logs(q,r,chi),p,energies)

    def _step(self,p,state,r,u,lr,key):
        chi=jax.vmap(coefficients)(u);e=self.energy(p,r,chi)[:,0]
        clipped=clip_training_energy(e)
        x,_=ravel_pytree(p)
        if self.name=='sr':
            def one(a,b):
                def logs(v):
                    z=self.model.logdata(self.unravel(v),MoireData(positions=a,spin_coords=b))
                    return jnp.stack([z.real,z.imag])
                return jax.jacrev(logs)(x)
            jac=microbatch_map(one,r,chi,8);o=jac[:,0,:]+1j*jac[:,1,:]
            direction=sr_direction(o,clipped,self.damping)
            delta=lr*direction
            metric=jnp.sqrt(jnp.mean(abs((o-o.mean(axis=0))@delta)**2))
        else:
            force=vmc_force(lambda q:self.logs(q,r,chi),p,clipped)
            scaled=jax.tree.map(lambda g:lr*g,force)
            updates,state=self.kfac.update(scaled,state,p,batched_data=self.batch(r,u),rngs=key)
            negative_update,_=ravel_pytree(updates);delta=-negative_update
            # Same empirical centered quantum-metric cap as SR, without its Jacobian.
            _,tangent=jax.jvp(lambda q:self.logs(q,r,chi),(p,),(self.unravel(delta),))
            metric=jnp.sqrt(jnp.mean(abs(tangent-tangent.mean())**2))
        delta*=jnp.minimum(1.,.05/jnp.maximum(metric,1e-12))
        delta*=jnp.minimum(1.,.15/jnp.maximum(jnp.linalg.norm(delta),1e-12))
        return self.unravel(x-delta),state,jnp.array([e.real.mean(),e.real.std(),e.imag.mean(),
                                                    jnp.linalg.norm(delta),metric])
