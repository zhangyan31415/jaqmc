# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields

import numpy as np
import pytest
from upath import UPath

from jaqmc.app.moire.model import MatrixContinuumModel


def _rotation(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray([[cosine, -sine], [sine, cosine]])


def _hex_q_vectors(basis: np.ndarray, radius: int) -> np.ndarray:
    return np.asarray(
        [
            np.asarray([i, j]) @ basis
            for i in range(-radius, radius + 1)
            for j in range(-radius, radius + 1)
            if max(abs(i), abs(j), abs(i + j)) <= radius
        ]
    )


def _zrs2_ab_model() -> MatrixContinuumModel:
    """Return the frozen 3.15-degree six-coefficient model as a generic model."""
    identity = np.eye(2)
    pauli_x = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    pauli_y = np.asarray([[0.0, -1j], [1j, 0.0]])
    pauli_z = np.diag([1.0, -1.0])
    internal_identity = np.eye(8)

    orbital_identity = np.kron(identity, identity)
    orbital_x = np.kron(pauli_x, identity)
    orbital_y_spin_z = np.kron(pauli_y, pauli_z)
    orbital_z = np.kron(pauli_z, identity)

    a1, a2, v1, v2, w1, w2 = (
        -18.33529132278916,
        -8.899214872791044,
        0.04480879088787322,
        -0.03822586965018016,
        0.019681497649847884,
        -0.01870262642798864,
    )
    momentum_powers = np.asarray([[0, 0], [2, 0], [1, 1], [0, 2]])
    momentum_matrices = np.stack(
        [
            v1 * np.kron(identity, orbital_z) + w1 * np.kron(pauli_x, orbital_identity),
            a1 * internal_identity + a2 * np.kron(identity, orbital_x),
            2.0 * a2 * np.kron(identity, orbital_y_spin_z),
            a1 * internal_identity - a2 * np.kron(identity, orbital_x),
        ]
    )

    first_vector = np.asarray([0.10865687010321617, 0.0])
    fourier_vectors = np.asarray(
        [_rotation(index * 2.0 * np.pi / 3.0) @ first_vector for index in range(3)]
    )
    orbital_matrices = (
        orbital_y_spin_z,
        np.sqrt(3.0) / 2.0 * orbital_x - 0.5 * orbital_y_spin_z,
        -np.sqrt(3.0) / 2.0 * orbital_x - 0.5 * orbital_y_spin_z,
    )
    fourier_matrices = np.stack(
        [
            v2 * np.kron(pauli_z, orbital_matrix)
            + w2 * np.kron(pauli_x, orbital_identity)
            for orbital_matrix in orbital_matrices
        ]
    )
    basis_labels = tuple(
        f"layer={layer},orbital={orbital},spin={spin}"
        for layer in range(2)
        for orbital in range(2)
        for spin in range(2)
    )
    return MatrixContinuumModel.from_fitted_terms(
        momentum_powers=momentum_powers,
        momentum_matrices=momentum_matrices,
        fourier_vectors=fourier_vectors,
        fourier_matrices=fourier_matrices,
        source_from_model_coordinates=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        basis_labels=basis_labels,
        name="ZrS2 AB 3.15 degree frozen fit",
        energy_unit="eV",
        length_unit="angstrom",
    )


def _zrs2_q_vectors() -> np.ndarray:
    reciprocal_length = 0.10865687010321617
    reciprocal_basis = np.asarray(
        [
            [reciprocal_length, 0.0],
            [reciprocal_length / 2.0, np.sqrt(3.0) * reciprocal_length / 2.0],
        ]
    )
    return MatrixContinuumModel.canonical_q_vectors(
        _hex_q_vectors(reciprocal_basis, radius=2),
        source_from_model_coordinates=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        plane_wave_q_sign=-1,
    )


def test_zrs2_fixture_uses_nineteen_q_vectors():
    q_vectors = _zrs2_q_vectors()

    assert q_vectors.shape == (19, 2)


def test_model_separates_constant_and_differential_terms():
    model = _zrs2_ab_model()

    np.testing.assert_array_equal(model.constant_matrix, model.momentum_matrices[0])
    powers, matrices = model.differential_terms
    np.testing.assert_array_equal(powers, model.momentum_powers[1:])
    np.testing.assert_array_equal(matrices, model.momentum_matrices[1:])


def test_model_converts_physical_units_to_dimensionless_coordinates():
    model = MatrixContinuumModel(
        momentum_powers=np.asarray([[0, 0], [1, 0], [0, 2]]),
        momentum_matrices=np.asarray(
            [
                [[1.0]],
                [[2.0]],
                [[3.0]],
            ]
        ),
        fourier_vectors=np.asarray([[0.2, -0.1]]),
        fourier_matrices=np.asarray([[[0.4]]]),
        energy_unit="eV",
        length_unit="angstrom",
    )

    converted = model.to_dimensionless(length_scale_nm=10.0)

    np.testing.assert_allclose(
        converted.momentum_matrices[:, 0, 0],
        np.asarray([1000.0, 20.0, 0.3]),
    )
    np.testing.assert_allclose(converted.fourier_vectors, [[20.0, -10.0]])
    np.testing.assert_allclose(converted.fourier_matrices, [[[400.0]]])
    assert converted.energy_unit == "meV"
    assert converted.length_unit == "dimensionless"


def test_fitted_term_import_transforms_coordinates_and_q_convention():
    angle = np.pi / 4.0
    source_from_model = _rotation(angle)
    matrix = np.asarray([[2.0]])

    model = MatrixContinuumModel.from_fitted_terms(
        momentum_powers=np.asarray([[2, 0]]),
        momentum_matrices=matrix[None, ...],
        fourier_vectors=np.asarray([[0.3, -0.2]]),
        fourier_matrices=np.asarray([[[0.4]]]),
        source_from_model_coordinates=source_from_model,
    )

    model_momentum = np.asarray([0.13, -0.27])
    source_momentum = source_from_model @ model_momentum
    expected_kinetic = source_momentum[0] ** 2 * matrix
    np.testing.assert_allclose(
        model.momentum_hamiltonian(model_momentum),
        expected_kinetic,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        model.fourier_vectors,
        np.asarray([[0.3, -0.2]]) @ source_from_model,
    )
    np.testing.assert_allclose(
        MatrixContinuumModel.canonical_q_vectors(
            np.asarray([[0.5, 0.1]]),
            source_from_model_coordinates=source_from_model,
            plane_wave_q_sign=-1,
        ),
        -np.asarray([[0.5, 0.1]]) @ source_from_model,
    )


@pytest.mark.parametrize(
    ("energy_unit", "length_unit", "message"),
    [
        ("kelvin", "angstrom", "Unsupported energy_unit"),
        ("meV", "meter", "Unsupported length_unit"),
    ],
)
def test_model_rejects_unsupported_units(energy_unit, length_unit, message):
    model = MatrixContinuumModel(
        momentum_powers=np.asarray([[0, 0]]),
        momentum_matrices=np.zeros((1, 1, 1)),
        fourier_vectors=np.zeros((0, 2)),
        fourier_matrices=np.zeros((0, 1, 1)),
        energy_unit=energy_unit,
        length_unit=length_unit,
    )

    with pytest.raises(ValueError, match=message):
        model.to_dimensionless(length_scale_nm=1.0)


@pytest.mark.parametrize(
    ("native_momentum", "expected_top_eigenvalues", "expected_trace"),
    [
        (
            [0.0, 0.0],
            [
                0.009595310546230145,
                0.009595310546230326,
                0.040142582037486656,
                0.040142582037486906,
                0.06341892496960001,
                0.06341892496960043,
                0.10334748468296061,
                0.10334748468296069,
            ],
            -83.12533735154926,
        ),
        (
            [0.013, -0.021],
            [
                0.014808686794938776,
                0.014808686794939012,
                0.03664912994472349,
                0.0366491299447239,
                0.05499657380154103,
                0.0549965738015411,
                0.09432100583491321,
                0.09432100583491342,
            ],
            -84.82538556299826,
        ),
        (
            [-0.037, 0.016],
            [
                0.02168573028677212,
                0.021685730286772377,
                0.038853225103376555,
                0.03885322510337667,
                0.04641825085033059,
                0.046418250850331036,
                0.080978479467314,
                0.0809784794673144,
            ],
            -87.65415430827815,
        ),
    ],
)
def test_generic_model_matches_frozen_zrs2_spectrum(
    native_momentum, expected_top_eigenvalues, expected_trace
):
    model = _zrs2_ab_model()

    hamiltonian = model.plane_wave_hamiltonian(
        np.asarray(native_momentum), _zrs2_q_vectors()
    )
    eigenvalues = np.linalg.eigvalsh(hamiltonian)

    np.testing.assert_allclose(
        eigenvalues[-8:], expected_top_eigenvalues, atol=2e-12, rtol=2e-12
    )
    np.testing.assert_allclose(eigenvalues.sum(), expected_trace, atol=2e-11)


def test_periodic_potential_is_hermitian_for_nonhermitian_fourier_coefficients():
    model = MatrixContinuumModel(
        momentum_powers=np.asarray([[0, 0]]),
        momentum_matrices=np.zeros((1, 2, 2)),
        fourier_vectors=np.asarray([[1.0, 0.0]]),
        fourier_matrices=np.asarray([[[0.0, 1.0 + 2.0j], [0.5j, 0.0]]]),
    )

    potential = model.periodic_potential(np.asarray([[0.2, 0.1], [0.4, -0.3]]))

    np.testing.assert_allclose(potential, potential.conj().transpose(0, 2, 1))


@pytest.mark.parametrize(
    "path_uri",
    ["{tmp_path}/model.npz", "memory://jaqmc-moire-model/model.npz"],
)
def test_matrix_continuum_model_roundtrip(tmp_path, path_uri):
    model = _zrs2_ab_model()
    path = UPath(path_uri.format(tmp_path=tmp_path))

    model.save(path)
    loaded = MatrixContinuumModel.load(path)

    for model_field in fields(model):
        expected = getattr(model, model_field.name)
        actual = getattr(loaded, model_field.name)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected


def test_matrix_continuum_model_rejects_opposite_fourier_vectors():
    with pytest.raises(ValueError, match=r"one vector from each \+/-g"):
        MatrixContinuumModel(
            momentum_powers=np.asarray([[0, 0]]),
            momentum_matrices=np.zeros((1, 1, 1)),
            fourier_vectors=np.asarray([[1.0, 0.0], [-1.0, 0.0]]),
            fourier_matrices=np.zeros((2, 1, 1)),
        )
