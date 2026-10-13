# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.app.moire.data import MoireData
from jaqmc.app.moire.matrix_hamiltonian import MatrixContinuumSecondMoment


def _measure(component_orbitals, amplitudes, matrices, normalize=False):
    def components(params, data):
        del params, data
        identity = jnp.einsum("ia,adij->dij", amplitudes, component_orbitals)
        return component_orbitals, amplitudes, identity

    n = amplitudes.shape[0]
    estimator = MatrixContinuumSecondMoment(
        f_internal_components=components,
        operator_matrices_real=matrices.real.tolist(),
        operator_matrices_imag=matrices.imag.tolist(),
        output_keys=[f"moment:{i}" for i in range(len(matrices))],
        normalize_by_electron=normalize,
        sum_output_key="moment:sum",
        sum_scale=0.25,
    )
    data = MoireData(positions=jnp.zeros((n, 2)), spin_coords=jnp.zeros(n))
    estimator.init(data, jax.random.PRNGKey(0))
    result, _ = estimator.evaluate_single_walker({}, data, {}, None, jax.random.PRNGKey(1))
    np.testing.assert_allclose(
        result["moment:sum"],
        sum(result[f"moment:{i}"] for i in range(len(matrices))) / 4,
        atol=2e-6,
    )
    return np.asarray([result[f"moment:{i}"] for i in range(len(matrices))])


def test_second_moment_matches_two_row_replacement_with_multiple_determinants():
    random = np.random.default_rng(72)
    components = jnp.asarray(
        random.normal(size=(3, 2, 3, 3)) + 1j * random.normal(size=(3, 2, 3, 3))
    )
    amplitudes = jnp.asarray(np.exp(1j * random.normal(size=(3, 3))))
    raw = random.normal(size=(2, 3, 3)) + 1j * random.normal(size=(2, 3, 3))
    matrices = jnp.asarray(raw + raw.conj().transpose(0, 2, 1))
    identity = np.einsum("ia,adij->dij", amplitudes, components)
    denominator = np.linalg.det(identity).sum()
    expected = []
    for matrix in np.asarray(matrices):
        transformed = np.einsum("ia,ab,bdij->dij", amplitudes, matrix, components)
        squared = np.einsum("ia,ab,bdij->dij", amplitudes, matrix @ matrix, components)
        value = 0j
        for i in range(3):
            one = identity.copy()
            one[:, i] = squared[:, i]
            value += np.linalg.det(one).sum() / denominator
            for k in range(3):
                if i == k:
                    continue
                pair = identity.copy()
                pair[:, i] = transformed[:, i]
                pair[:, k] = transformed[:, k]
                value += np.linalg.det(pair).sum() / denominator
        expected.append(value)
    np.testing.assert_allclose(
        _measure(components, amplitudes, matrices), expected, rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        _measure(components, amplitudes, matrices, True),
        np.asarray(expected) / 9,
        rtol=2e-5,
        atol=2e-5,
    )


def test_two_spin_singlet_and_saturated_triplet_have_correct_s_squared():
    matrices = jnp.asarray(
        [
            [[0, 1], [1, 0]],
            [[0, -1j], [1j, 0]],
            [[1, 0], [0, -1]],
        ]
    )
    amplitudes = jnp.asarray([[1, jnp.exp(0.3j)], [1, jnp.exp(1.2j)]])
    # Shared spatial orbital with opposite spin columns is the exact singlet.
    singlet = jnp.asarray([[[[1, 0], [1, 0]]], [[[0, 1], [0, 1]]]], dtype=complex)
    result = _measure(singlet, amplitudes, matrices)
    np.testing.assert_allclose(result, np.zeros(3), atol=2e-6)
    # Two orthogonal spatial orbitals, both spin up, form |S=1, Sz=1>.
    triplet = jnp.asarray([[[[1, 0], [0, 1]]], [[[0, 0], [0, 0]]]], dtype=complex)
    result = _measure(triplet, amplitudes, matrices)
    # Transverse local estimators contain phase-dependent double spin flips.
    # Uniform internal-angle integration removes these terms; their sum
    # cancels pointwise in S squared.
    phase = 2 * np.exp(1.5j)
    np.testing.assert_allclose(result, [2 + phase, 2 - phase, 4], atol=2e-6)
    np.testing.assert_allclose(result.sum() / 4, 2, atol=2e-6)
    opposite_phase = amplitudes.at[1, 1].multiply(-1)
    averaged = (result + _measure(triplet, opposite_phase, matrices)) / 2
    np.testing.assert_allclose(averaged, [2, 2, 4], atol=2e-6)
