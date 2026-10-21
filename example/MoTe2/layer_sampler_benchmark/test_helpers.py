"""Numerical contracts independent of a GPU or trained checkpoint."""
import unittest
import numpy as np
from helpers import sr_direction,microbatch_map,jnp


class NumericalContracts(unittest.TestCase):
    def test_sample_space_sr_matches_parameter_space_metric(self):
        rng=np.random.default_rng(17)
        o=rng.normal(size=(9,7))+1j*rng.normal(size=(9,7))
        e=rng.normal(size=9)+1j*rng.normal(size=9)
        centered=o-o.mean(axis=0);de=e-e.mean()
        metric=(centered.conj().T@centered).real/9+.13*np.eye(7)
        force=(centered.conj().T@de).real/9
        np.testing.assert_allclose(sr_direction(jnp.asarray(o),jnp.asarray(e),.13),
                                   np.linalg.solve(metric,force),rtol=1e-10,atol=1e-10)

    def test_microbatch_preserves_walker_order_and_outputs(self):
        r=jnp.arange(24,dtype=jnp.float64).reshape(6,2,2)
        u=jnp.arange(12,dtype=jnp.float64).reshape(6,2)
        actual=microbatch_map(lambda a,b:{'sum':a.sum()+b.sum(),'row':a[0]},r,u,2)
        np.testing.assert_array_equal(actual['sum'],r.sum(axis=(1,2))+u.sum(axis=1))
        np.testing.assert_array_equal(actual['row'],r[:,0])
        with self.assertRaises(ValueError):microbatch_map(lambda a,b:a,r,u,4)

if __name__=='__main__':unittest.main()
