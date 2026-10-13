# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.moire.data import MoireData
from jaqmc.app.moire.matrix_hamiltonian import (
    MatrixContinuumDifferential,
    MatrixContinuumPotential,
)


def _component_fixture():
    component_orbitals = jnp.asarray(
        [
            [[[1.2 + 0.1j, 0.2], [0.1j, 0.9 - 0.1j]]],
            [[[0.3 - 0.2j, 0.1j], [0.25, -0.15 + 0.05j]]],
        ]
    )

    def components(params, data):
        del params
        amplitudes = jnp.stack(
            [
                jnp.ones_like(data.spin_coords, dtype=complex),
                jnp.exp(1j * data.spin_coords),
            ],
            axis=-1,
        )
        identity = jnp.einsum("ia,adij->dij", amplitudes, component_orbitals)
        return component_orbitals, amplitudes, identity

    return component_orbitals, components


def test_matrix_continuum_potential_matches_direct_row_replacement():
    component_orbitals, components = _component_fixture()
    data = MoireData(
        positions=jnp.asarray([[0.2, -0.1], [0.45, 0.3]]),
        spin_coords=jnp.asarray([0.25, 1.1]),
    )
    constant = jnp.asarray([[0.4, 0.1 - 0.2j], [0.1 + 0.2j, -0.3]])
    vectors = jnp.asarray([[1.2, -0.4]])
    coefficients = jnp.asarray([[[0.05j, 0.2], [-0.1j, 0.07 - 0.03j]]])
    estimator = MatrixContinuumPotential(
        f_internal_components=components,
        constant_matrix=constant,
        fourier_vectors=vectors,
        fourier_matrices=coefficients,
    )
    estimator.init(data, jax.random.PRNGKey(0))

    stats, _ = estimator.evaluate_single_walker(
        {}, data, {}, None, jax.random.PRNGKey(1)
    )

    _, amplitudes, identity = components({}, data)
    phase = jnp.exp(1j * (data.positions @ vectors.T))
    forward = jnp.einsum("ig,gab->iab", phase, coefficients)
    potential = constant[None, ...] + forward + jnp.swapaxes(forward.conj(), -1, -2)
    psi = jnp.linalg.det(identity[0])
    expected = 0.0j
    for electron in range(2):
        transformed_row = jnp.einsum(
            "a,ab,bj->j",
            amplitudes[electron],
            potential[electron],
            component_orbitals[:, 0, electron],
        )
        numerator = jnp.linalg.det(identity[0].at[electron].set(transformed_row))
        expected += numerator / psi

    np.testing.assert_allclose(
        stats["energy:matrix_potential"], expected, rtol=1e-6, atol=1e-6
    )


def test_matrix_continuum_potential_rejects_component_mismatch():
    _, components = _component_fixture()
    data = MoireData(
        positions=jnp.zeros((2, 2)),
        spin_coords=jnp.zeros((2,)),
    )
    estimator = MatrixContinuumPotential(
        f_internal_components=components,
        constant_matrix=jnp.eye(3),
        fourier_vectors=jnp.zeros((0, 2)),
        fourier_matrices=jnp.zeros((0, 3, 3)),
    )
    estimator.init(data, jax.random.PRNGKey(0))

    with pytest.raises(ValueError, match="component count"):
        estimator.evaluate_single_walker({}, data, {}, None, jax.random.PRNGKey(1))


def _differential_components(params, data):
    del params
    x, y = data.positions.T
    component_0 = jnp.asarray(
        [
            [
                1.1 + 0.2 * x[0] + 0.03 * y[0] ** 2 + 0.05j * x[0] * y[0],
                0.2 + 0.1 * x[0] + 0.04 * jnp.sum(x),
            ],
            [
                0.15 + 0.08 * y[1] + 0.03j * x[0] * y[1],
                0.9 + 0.12 * x[1] ** 2 - 0.04j * y[1],
            ],
        ]
    )
    component_1 = jnp.asarray(
        [
            [
                0.3 - 0.06 * y[0] + 0.02j * x[1],
                -0.1 + 0.07 * x[0] * y[0] + 0.03j * y[1] ** 2,
            ],
            [
                0.25 + 0.05 * x[1] + 0.02 * jnp.sum(y),
                -0.2 + 0.09 * x[1] * y[1] + 0.04j * x[1] ** 2,
            ],
        ]
    )
    component_orbitals = jnp.stack([component_0, component_1])[:, None, ...]
    amplitudes = jnp.stack(
        [
            jnp.exp(1j * data.spin_coords),
            jnp.exp(-1j * data.spin_coords),
        ],
        axis=-1,
    )
    identity = jnp.einsum("ia,adij->dij", amplitudes, component_orbitals)
    return component_orbitals, amplitudes, identity


def test_matrix_continuum_differential_matches_direct_autodiff_oracle():
    data = MoireData(
        positions=jnp.asarray([[0.2, -0.1], [0.45, 0.3]]),
        spin_coords=jnp.asarray([0.25, 1.1]),
    )
    powers = jnp.asarray([[1, 0], [0, 1], [2, 0], [1, 1], [0, 2]])
    matrices = jnp.asarray(
        [
            [[0.3, 0.1 + 0.05j], [0.1 - 0.05j, -0.2]],
            [[-0.15, 0.04j], [-0.04j, 0.25]],
            [[0.6, -0.08 + 0.02j], [-0.08 - 0.02j, 0.1]],
            [[0.12, 0.03 - 0.06j], [0.03 + 0.06j, -0.18]],
            [[-0.2, 0.07], [0.07, 0.45]],
        ]
    )
    estimator = MatrixContinuumDifferential(
        f_internal_components=_differential_components,
        momentum_powers=powers,
        momentum_matrices=matrices,
    )
    estimator.init(data, jax.random.PRNGKey(0))

    stats, _ = estimator.evaluate_single_walker(
        {}, data, {}, None, jax.random.PRNGKey(1)
    )

    _, _, identity = _differential_components({}, data)
    psi = jnp.linalg.det(identity[0])
    flat_positions = data.positions.reshape(-1)
    expected = 0.0j
    for power, matrix in zip(powers.tolist(), matrices):
        degree = sum(power)
        for electron_idx in range(len(data.positions)):

            def numerator(spatial_flat):
                walker_data = data.merge(
                    {"positions": spatial_flat.reshape(data.positions.shape)}
                )
                orbitals, amplitudes, base = _differential_components({}, walker_data)
                transformed_row = jnp.einsum(
                    "a,ab,bj->j",
                    amplitudes[electron_idx],
                    matrix,
                    orbitals[:, 0, electron_idx],
                )
                return jnp.linalg.det(base[0].at[electron_idx].set(transformed_row))

            first_axis = 0 if power[0] else 1
            first_coord = 2 * electron_idx + first_axis
            if degree == 1:
                derivative = jax.jacfwd(numerator)(flat_positions)[first_coord]
            else:
                second_axis = 1 if power[1] == 2 else first_axis
                if power == [1, 1]:
                    second_axis = 1
                second_coord = 2 * electron_idx + second_axis
                derivative = jax.jacfwd(jax.jacfwd(numerator))(flat_positions)[
                    first_coord, second_coord
                ]
            expected += (-1j) ** degree * derivative / psi

    np.testing.assert_allclose(
        stats["energy:matrix_differential"], expected, rtol=2e-5, atol=2e-5
    )
    compiled_energy = jax.jit(
        lambda positions: estimator.evaluate_single_walker(
            {},
            data.merge({"positions": positions}),
            {},
            None,
            jax.random.PRNGKey(2),
        )[0]["energy:matrix_differential"]
    )(data.positions)
    np.testing.assert_allclose(compiled_energy, expected, rtol=2e-5, atol=2e-5)


def test_matrix_continuum_differential_rejects_unsupported_order():
    data = MoireData(
        positions=jnp.zeros((2, 2)),
        spin_coords=jnp.zeros((2,)),
    )
    estimator = MatrixContinuumDifferential(
        f_internal_components=_differential_components,
        momentum_powers=jnp.asarray([[3, 0]]),
        momentum_matrices=jnp.zeros((1, 2, 2)),
    )

    with pytest.raises(ValueError, match="total degree one or two"):
        estimator.init(data, jax.random.PRNGKey(0))
