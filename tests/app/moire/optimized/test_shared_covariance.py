# Copyright (c) 2026 JaQMC contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent covariance identity for the weighted reverse pass."""

from operator import mul
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from jaqmc.app.moire.optimized.shared import WeightedLossAndGrad


def test_weighted_gradient_matches_explicit_complex_covariance():
    x = jnp.array([[0.2, 0.8], [-0.4, 1.1], [0.7, -0.3], [1.2, 0.9]])
    params = jnp.array([0.3, -0.6])
    energy = jnp.array([1.3 + 0.7j, -0.8 + 0.1j, 2.1 - 0.3j, 0.5 - 0.4j])

    def logpsi(p, datum):
        return jnp.dot(p, datum) + 1j * jnp.dot(p**2, datum**2)

    estimator = WeightedLossAndGrad(f_log_psi=logpsi, clip_method="none")
    actual, state = estimator.evaluate_batch_walkers(
        params,
        SimpleNamespace(data=x, vmap_axis=0),
        {"total_energy": energy},
        None,
        None,
    )
    jacobian = x + 2j * params * x**2
    expected = 2 * jnp.real(
        jnp.mean(
            jnp.conj(jacobian - jacobian.mean(axis=0))
            * (energy - energy.mean())[:, None],
            axis=0,
        )
    )
    np.testing.assert_allclose(actual["grads"], expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual["loss"], energy.mean())
    assert state is None


def test_weighted_gradient_rejects_complex_parameters():
    import pytest

    estimator = WeightedLossAndGrad(f_log_psi=mul, clip_method="none")
    with pytest.raises(ValueError, match="real network parameters"):
        estimator.evaluate_batch_walkers(
            jnp.array(1 + 1j),
            SimpleNamespace(data=jnp.ones(2), vmap_axis=0),
            {"total_energy": jnp.ones(2)},
            None,
            jax.random.key(0),
        )
