"""Complex curvature and VMC-force contracts for the optional KFAC backend."""
import unittest
import importlib.util
import numpy as np
from helpers import jax,jnp

@unittest.skipUnless(importlib.util.find_spec('kfac_jax'),'Install requirements-kfac.txt to test KFAC')
class ComplexKFAC(unittest.TestCase):
    def test_unregistered_shared_phase_parameter_is_rejected(self):
        from nqs18 import MoireData
        from jaqmc.data import BatchedData
        from jaqmc.optimizer.kfac import KFACOptimizer
        r=jnp.array([[[-2.,1.]], [[-1.,-1.]], [[1.,-1.]], [[2.,1.]]])
        data=BatchedData(MoireData(positions=r,spin_coords=jnp.zeros((4,1))),['positions','spin_coords'])
        def logpsi(p,d):return p['a']*d.positions[0,0]+1j*p['b']*d.positions[0,1]
        opt=KFACOptimizer(learning_rate=.01,f_log_psi=logpsi,pmap_axis_name=None,reject_generic=True)
        with self.assertRaisesRegex(ValueError,'generic'):
            opt.init({'a':jnp.array(.2),'b':jnp.array(.3)},batched_data=data,rngs=jax.random.PRNGKey(5))

    def test_float64_repeated_dense_bias_is_not_generic(self):
        from nqs18 import MoireData
        from jaqmc.data import BatchedData
        from jaqmc.optimizer.kfac import KFACOptimizer
        import flax.linen as nn
        layer=nn.Dense(3,param_dtype=jnp.float64)
        r=jnp.arange(16,dtype=jnp.float64).reshape(4,2,2)
        p=layer.init(jax.random.PRNGKey(3),r[0])
        data=BatchedData(MoireData(positions=r,spin_coords=jnp.zeros((4,2))),['positions','spin_coords'])
        def logpsi(p,d):
            h=layer.apply(p,d.positions)
            return h.sum()+1j*(h*h).sum()
        opt=KFACOptimizer(learning_rate=.01,f_log_psi=logpsi,pmap_axis_name=None)
        opt.init(p,batched_data=data,rngs=jax.random.PRNGKey(4))
        variants=[eq.params['meta'].variant for eq in opt._preconditioner.estimator.jaxpr.layer_tags]
        self.assertNotIn('generic',variants)

    def test_vmc_force_includes_imaginary_energy_and_phase(self):
        from optimizers import vmc_force
        x=jnp.array([-2.,-1.,1.,2.]);y=jnp.array([1.,-1.,-1.,1.])
        p={'a':jnp.array(.2),'b':jnp.array(.3)}
        gradient=vmc_force(lambda q:q['a']*x+1j*q['b']*y,p,x+2j*y)
        np.testing.assert_allclose([gradient['a'],gradient['b']],[2.5,2.],atol=1e-12)

    def test_phase_and_amplitude_curvature(self):
        from nqs18 import MoireData
        from jaqmc.data import BatchedData
        from jaqmc.optimizer.kfac import KFACOptimizer
        r=jnp.array([[[-2.,1.]], [[-1.,-1.]], [[1.,-1.]], [[2.,1.]]])
        data=BatchedData(MoireData(positions=r,spin_coords=jnp.zeros((4,1))),['positions','spin_coords'])
        # Match the model's real amplitude/phase heads before complex assembly.
        def logpsi(p,d):return p['a']*d.positions[0,0]+1j*(p['b']*d.positions[0,1])
        p={'a':jnp.array(.2),'b':jnp.array(.3)}
        optimizer=KFACOptimizer(learning_rate=.01,f_log_psi=logpsi,pmap_axis_name=None,
            predictive_variance=1.,norm_constraint=None,damping=.05)
        state=optimizer.init(p,batched_data=data,rngs=jax.random.PRNGKey(1))
        pre=optimizer._preconditioner
        update=jax.jit(lambda s:pre.maybe_update_estimator_curvature(s,(p,data),jax.random.PRNGKey(2),sync=False))
        state=update(state.kfac)
        matrix=np.asarray(pre.estimator.to_dense_matrix(state.estimator_state))
        np.testing.assert_allclose(matrix,np.diag([2.5,1.]),rtol=1e-9,atol=1e-9)
        self.assertGreater(matrix[1,1],0.,'Phase tangent must contribute to the quantum metric')

if __name__=='__main__':unittest.main()
