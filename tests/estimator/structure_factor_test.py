# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.moire.data import MoireData
from jaqmc.estimator.structure_factor import PeriodicStructureFactor


def test_periodic_structure_factor_matches_direct_sum():
    data = MoireData(
        positions=jnp.asarray([[0.1, 0.2], [0.45, 0.7], [0.8, 0.35]]),
        spin_coords=jnp.zeros((3,)),
    )
    lattice = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    indices = [[1, 0], [1, 1], [-1, 2]]
    estimator = PeriodicStructureFactor(
        reciprocal_indices=indices,
        lattice=lattice,
    )
    estimator.init(data, jax.random.PRNGKey(0))

    stats, _ = estimator.evaluate_single_walker(
        {}, data, {}, None, jax.random.PRNGKey(1)
    )

    q_vectors = 2.0 * np.pi * np.asarray(indices)
    density = np.exp(1j * (np.asarray(data.positions) @ q_vectors.T)).sum(axis=0)
    np.testing.assert_allclose(
        stats["observable:density_mode"],
        density / len(data.positions),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        stats["observable:structure_factor"],
        np.abs(density) ** 2 / len(data.positions),
        rtol=1e-6,
        atol=1e-6,
    )


def test_periodic_structure_factor_is_translation_invariant():
    data = MoireData(
        positions=jnp.asarray([[0.1, 0.2], [0.45, 0.7], [0.8, 0.35]]),
        spin_coords=jnp.zeros((3,)),
    )
    estimator = PeriodicStructureFactor(
        reciprocal_indices=[[1, 0], [1, 1]],
        lattice=jnp.asarray([[2.0, 0.0], [0.0, 3.0]]),
    )
    estimator.init(data, jax.random.PRNGKey(0))
    base, _ = estimator.evaluate_single_walker(
        {}, data, {}, None, jax.random.PRNGKey(1)
    )
    translated, _ = estimator.evaluate_single_walker(
        {},
        data.merge({"positions": data.positions + jnp.asarray([0.37, -0.21])}),
        {},
        None,
        jax.random.PRNGKey(2),
    )

    np.testing.assert_allclose(
        translated["observable:structure_factor"],
        base["observable:structure_factor"],
        rtol=1e-6,
        atol=1e-6,
    )
    assert not np.allclose(
        translated["observable:density_mode"],
        base["observable:density_mode"],
    )


def test_periodic_structure_factor_rejects_zero_vector():
    data = MoireData(
        positions=jnp.zeros((2, 2)),
        spin_coords=jnp.zeros((2,)),
    )
    estimator = PeriodicStructureFactor(
        reciprocal_indices=[[0, 0]],
        lattice=jnp.eye(2),
    )

    with pytest.raises(ValueError, match="zero vector"):
        estimator.init(data, jax.random.PRNGKey(0))
