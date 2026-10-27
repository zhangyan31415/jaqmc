# Copyright (c) 2026 JaQMC contributors
# SPDX-License-Identifier: Apache-2.0

"""Layer-basis equivalence and independent binary heat-bath checks."""

from itertools import product
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaqmc.app.moire.data import MoireData
from jaqmc.app.moire.optimized import mote2
from jaqmc.app.moire.optimized.discrete import _discrete_components, validate_labels
from jaqmc.app.moire.optimized.sampling import (
    conditional_probability,
    gibbs_sweep,
    selected_matrix,
)
from jaqmc.app.moire.wavefunction import MoireWavefunction


def test_continuous_fourier_norm_equals_sum_over_discrete_layers():
    # This identity includes electron-dependent backflow matrices: no factorized
    # one-particle spatial assumption enters the layer Fourier transformation.
    rng = np.random.default_rng(8)
    n = 3
    phi = rng.normal(size=(2, 1, n, n)) + 1j * rng.normal(size=(2, 1, n, n))
    fake = SimpleNamespace(
        nlayers=2,
        ndets=1,
        phase_mode="singlephase",
        _get_singlephase_layer_orbitals=lambda data: tuple(phi),
    )
    discrete_norm = 0.0
    for labels in product((0.0, 1.0), repeat=n):
        data = MoireData(positions=jnp.zeros((n, 2)), spin_coords=jnp.asarray(labels))
        _, _, selected = _discrete_components(fake, data)
        discrete_norm += abs(np.linalg.det(np.asarray(selected)[0])) ** 2
    continuous_norm = 0.0
    for phase_indices in product(range(3), repeat=n):
        angles = np.asarray(phase_indices) * (2 * np.pi / 3)
        chi = np.stack((np.exp(1j * angles), np.exp(-1j * angles)), axis=-1)
        matrix = np.einsum("il,ldij->dij", chi, phi)[0]
        continuous_norm += abs(np.linalg.det(matrix)) ** 2 / 3**n
    np.testing.assert_allclose(discrete_norm, continuous_norm, rtol=2e-6)


def test_conditional_probability_matches_two_full_determinants():
    rng = np.random.default_rng(13)
    phi = jnp.asarray(
        rng.normal(size=(5, 2, 1, 3, 3)) + 1j * rng.normal(size=(5, 2, 1, 3, 3))
    )
    labels = jnp.asarray([[0.0, 1.0, 0.0]] * 5)
    inverse = jnp.linalg.inv(selected_matrix(phi, labels))
    for i in range(3):
        actual, valid = conditional_probability(phi, inverse, i)
        weights = []
        for value in (0.0, 1.0):
            matrix = selected_matrix(phi, labels.at[:, i].set(value))
            weights.append(jnp.abs(jnp.linalg.det(matrix)) ** 2)
        expected = weights[0] / (weights[0] + weights[1])
        assert np.asarray(valid).all()
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-7)
    result, stats = jax.jit(gibbs_sweep)(phi, labels, jax.random.key(4))
    validate_labels(result)
    assert int(stats["spin_nonfinite_count"]) == 0
    assert float(stats["spin_inverse_residual_max"]) < 1e-3


def test_representation_selection_restores_native_method():
    native = MoireWavefunction.get_layer_components
    try:
        mote2.prepare_representation("discrete")
        assert MoireWavefunction.get_layer_components is _discrete_components
        mote2.prepare_representation("continuous")
        assert MoireWavefunction.get_layer_components is native
        with pytest.raises(ValueError, match="representation"):
            mote2.prepare_representation("unknown")
    finally:
        mote2.prepare_representation("continuous")


def test_sparse_guards_and_invalid_labels_fail_explicitly():
    fake = SimpleNamespace(nlayers=2, ndets=1, phase_mode="singlephase")
    with pytest.raises(ValueError, match="chunk=512"):
        mote2.install_backend(None, fake, chunk=64)
    with pytest.raises(ValueError, match="0/1"):
        validate_labels(np.asarray([0.0, 0.3]))
