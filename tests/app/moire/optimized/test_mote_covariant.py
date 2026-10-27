# Copyright (c) 2026 JaQMC contributors
# SPDX-License-Identifier: Apache-2.0

"""Analytic valley-shift identity independent of native operator code."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaqmc.app.moire.data import MoireData
from jaqmc.app.moire.optimized.covariant import CovariantKinetic
from jaqmc.app.moire.optimized.discrete import DiscreteCovariantKinetic


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("discrete", [False, True])
def test_layer_plane_waves_have_expected_covariant_energy(sparse, discrete):
    lattice = np.array([[1.8, 0.2], [0.3, 1.4]])
    # reciprocal rows obey A @ B.T = 2pi I; derive momenta independently.
    reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
    kb = reciprocal.sum(axis=0) / 3
    kt = (2 * reciprocal[0] - reciprocal[1]) / 3
    labels = np.array([0, 1])
    eta = np.array([1, -1])
    exponent = 1 - 2 * labels
    c = np.array([[0.2 + 0.1j, -0.3 + 0.4j], [0.5 - 0.2j, -0.1 + 0.3j]])
    shifted_gradient = c - 1j * eta[:, None] * np.stack((kb, kt))
    expected = -0.5 * 0.7 * np.sum(shifted_gradient**2)

    def logpsi(params, data):
        del params
        value = jnp.sum(jnp.asarray(c) * data.positions)
        return (
            value
            if discrete
            else value + 1j * jnp.sum(jnp.asarray(exponent) * data.spin_coords)
        )

    data = MoireData(
        positions=jnp.array([[0.3, 0.7], [0.8, 0.1]]),
        spin_coords=jnp.asarray(labels if discrete else [0.2, 0.9], dtype=jnp.float32),
    )
    if sparse:
        from jaqmc.app.moire.optimized.folx_compat import install

        install()
    cls = DiscreteCovariantKinetic if discrete else CovariantKinetic
    estimator = cls(
        f_log_psi=logpsi,
        primitive_lattice=jnp.asarray(lattice),
        nspins=(1, 1),
        prefactor=0.7,
        sparse=sparse,
    )
    answer, state = estimator.evaluate_single_walker(None, data, {}, None, None)
    np.testing.assert_allclose(
        answer["energy:kinetic_plus_soc"], expected, rtol=2e-6, atol=2e-6
    )
    assert state is None
