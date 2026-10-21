# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Physics gates for the two-shell MoTe2 continuum Hamiltonian."""

import numpy as np
from jax import numpy as jnp

from jaqmc.app.moire.free_reference import free_fermion_reference
from jaqmc.app.moire.hamiltonian import (
    _moire_first_shell_vectors,
    _moire_potential_matrix,
    _moire_valley_momenta,
)
from jaqmc.app.moire.model import MatrixContinuumModel

LATTICE = np.asarray([[np.sqrt(3.0) / 2.0, -0.5], [0.0, 1.0]], dtype=float)

# Exact 3.890238169-degree values from
# MoTe2/AA/models/K/six_parameter_joint_20261016/cases/8_3.89/model.npz
# at source commit 162cceb54ebae50c2dd434965f4974580202a4aa.
PORTABLE_PARAMETERS_EV = np.asarray(
    [
        5.608008887416021,
        0.000204443854088502,
        0.009295456371455268,
        -0.010228764449594223,
        0.015888448347906002,
        0.00946347593311961,
    ]
)
PORTABLE_RECIPROCAL_PER_ANGSTROM = np.asarray(
    [
        [0.13820712168748003, 0.0],
        [0.06910356084374002, 0.11969087836528494],
    ]
)
PORTABLE_KAPPAS_PER_ANGSTROM = np.asarray(
    [
        [0.06910356084375123, -0.039896959454429844],
        [0.06910356084376174, 0.03989695945642489],
    ]
)


def _unit_cell_grid(lattice: np.ndarray, size: int = 12) -> np.ndarray:
    fractions = np.asarray(
        [(i / size, j / size) for i in range(size) for j in range(size)]
    )
    return fractions @ np.asarray(lattice)


def _reciprocal_vectors(lattice: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * np.linalg.inv(np.asarray(lattice)).T


def _q_grid(reciprocal: np.ndarray, cutoff: int) -> np.ndarray:
    indices = np.asarray(
        [(i, j) for i in range(-cutoff, cutoff + 1) for j in range(-cutoff, cutoff + 1)]
    )
    return indices @ reciprocal


def _implementation_hamiltonian(
    momentum: np.ndarray,
    q_vectors: np.ndarray,
    *,
    lattice: np.ndarray,
    spin_sign: int,
    kinetic_coefficient_mev: float,
    v1_mev: float,
    phi1_rad: float,
    omega1_mev: float,
    v2_mev: float,
    omega2_mev: float,
) -> np.ndarray:
    """Build the plane-wave matrix from the implemented real-space field."""
    positions = _unit_cell_grid(lattice)
    potential = np.asarray(
        _moire_potential_matrix(
            jnp.asarray(positions),
            jnp.asarray(lattice),
            jnp.full((len(positions),), spin_sign),
            v1_mev=v1_mev,
            phi1_rad=phi1_rad,
            omega1_mev=omega1_mev,
            v2_mev=v2_mev,
            omega2_mev=omega2_mev,
        )
    )
    differences = q_vectors[:, None, :] - q_vectors[None, :, :]
    phases = np.exp(-1j * np.einsum("rd,ijd->rij", positions, differences))
    nq = len(q_vectors)
    matrix = np.zeros((2, nq, 2, nq), dtype=complex)
    for layer_i in range(2):
        for layer_j in range(2):
            matrix[layer_i, :, layer_j, :] = np.einsum(
                "rij,r->ij",
                phases,
                potential[:, layer_i, layer_j],
                optimize=True,
            ) / len(positions)

    k_plus, k_minus = _moire_valley_momenta(jnp.asarray(lattice))
    valleys = spin_sign * np.asarray([k_plus, k_minus])
    for layer in range(2):
        kinetic = kinetic_coefficient_mev * np.sum(
            (momentum[None, :] + q_vectors - valleys[layer]) ** 2,
            axis=-1,
        )
        matrix[layer, :, layer, :] += np.diag(kinetic)
    return matrix.reshape(2 * nq, 2 * nq)


def _portable_extended_response(
    momentum: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    reciprocal: np.ndarray,
    kappas: np.ndarray,
) -> np.ndarray:
    """Independent transcription of the portable model.py coefficient basis."""
    n1, n2 = len(q1), len(q2)
    n = n1 + n2
    q = np.vstack([q1, q2])
    output = np.zeros((6, n, n), dtype=complex)
    output[0, np.arange(n), np.arange(n)] = -np.sum(
        (momentum[None, :] - q) ** 2, axis=-1
    )

    first_star = np.asarray(
        [reciprocal[0], reciprocal[1] - reciprocal[0], -reciprocal[1]]
    )
    for layer, layer_q in enumerate((q1, q2)):
        difference = layer_q[:, None, :] - layer_q[None, :, :]
        plus = np.sum(
            np.linalg.norm(difference[:, :, None] - first_star, axis=-1) < 1e-8,
            axis=-1,
        )
        minus = np.sum(
            np.linalg.norm(difference[:, :, None] + first_star, axis=-1) < 1e-8,
            axis=-1,
        )
        layer_slice = slice(0, n1) if layer == 0 else slice(n1, n)
        output[1, layer_slice, layer_slice] = plus + minus
        output[2, layer_slice, layer_slice] = (-1j if layer == 0 else 1j) * (
            plus - minus
        )

    delta = kappas[0] - kappas[1]
    nearest = np.asarray(
        [delta, delta + reciprocal[1], delta + reciprocal[1] - reciprocal[0]]
    )
    interlayer_difference = q1[:, None, :] - q2[None, :, :]
    first_tunneling = np.sum(
        np.linalg.norm(
            interlayer_difference[:, :, None] - nearest,
            axis=-1,
        )
        < 1e-8,
        axis=-1,
    )
    output[3, :n1, n1:] = -first_tunneling
    output[3, n1:, :n1] = -first_tunneling.T

    second_star = np.asarray(
        [
            reciprocal[0] + reciprocal[1],
            reciprocal[1] - 2.0 * reciprocal[0],
            reciprocal[0] - 2.0 * reciprocal[1],
        ]
    )
    for layer, layer_q in enumerate((q1, q2)):
        difference = layer_q[:, None, :] - layer_q[None, :, :]
        second_intralayer = np.sum(
            np.linalg.norm(difference[:, :, None] - second_star, axis=-1) < 1e-8,
            axis=-1,
        ) + np.sum(
            np.linalg.norm(difference[:, :, None] + second_star, axis=-1) < 1e-8,
            axis=-1,
        )
        layer_slice = slice(0, n1) if layer == 0 else slice(n1, n)
        output[4, layer_slice, layer_slice] = second_intralayer

    second_tunneling = np.sum(
        np.linalg.norm(
            interlayer_difference[:, :, None] + 2.0 * nearest,
            axis=-1,
        )
        < 1e-8,
        axis=-1,
    )
    output[5, :n1, n1:] = -second_tunneling
    output[5, n1:, :n1] = -second_tunneling.T
    return output


def _portable_hole_hamiltonian(
    momentum: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    parameters_mev: np.ndarray,
) -> tuple[np.ndarray, float]:
    def electron(kpoint: np.ndarray) -> np.ndarray:
        response = _portable_extended_response(
            kpoint,
            q1,
            q2,
            PORTABLE_RECIPROCAL_PER_ANGSTROM,
            PORTABLE_KAPPAS_PER_ANGSTROM,
        )
        return np.einsum("pij,p->ij", response, parameters_mev, optimize=True)

    top = float(np.linalg.eigvalsh(electron(np.zeros(2)))[-1])
    electron_at_minus_k = electron(-np.asarray(momentum))
    return top * np.eye(len(electron_at_minus_k)) - electron_at_minus_k.T, top


def _reference_model_from_implementation(
    *,
    lattice: np.ndarray,
    kinetic_coefficient_mev: float,
    v1_mev: float,
    phi1_rad: float,
    omega1_mev: float,
    v2_mev: float,
    omega2_mev: float,
) -> MatrixContinuumModel:
    """Fourier transform the implemented field into the canonical model format."""
    reciprocal = _reciprocal_vectors(lattice)
    positions = _unit_cell_grid(lattice)
    potential = np.asarray(
        _moire_potential_matrix(
            jnp.asarray(positions),
            jnp.asarray(lattice),
            jnp.ones((len(positions),)),
            v1_mev=v1_mev,
            phi1_rad=phi1_rad,
            omega1_mev=omega1_mev,
            v2_mev=v2_mev,
            omega2_mev=omega2_mev,
        )
    )

    def coefficient(index: tuple[int, int]) -> np.ndarray:
        vector = np.asarray(index) @ reciprocal
        phase = np.exp(-1j * (positions @ vector))
        return np.einsum("r,rij->ij", phase, potential) / len(positions)

    k_plus, k_minus = _moire_valley_momenta(jnp.asarray(lattice))
    valleys = np.asarray([k_plus, k_minus])
    identity = np.eye(2)
    constant = kinetic_coefficient_mev * np.diag(
        np.sum(valleys**2, axis=-1)
    ) + coefficient((0, 0))
    momentum_matrices = np.stack(
        [
            constant,
            -2.0 * kinetic_coefficient_mev * np.diag(valleys[:, 0]),
            -2.0 * kinetic_coefficient_mev * np.diag(valleys[:, 1]),
            kinetic_coefficient_mev * identity,
            kinetic_coefficient_mev * identity,
        ]
    )

    indices = [
        (i, j)
        for i in range(-2, 3)
        for j in range(-2, 3)
        if i > 0 or (i == 0 and j > 0)
    ]
    terms = [(np.asarray(index) @ reciprocal, coefficient(index)) for index in indices]
    terms = [
        (vector, matrix) for vector, matrix in terms if np.max(np.abs(matrix)) > 1e-4
    ]
    return MatrixContinuumModel(
        momentum_powers=np.asarray([[0, 0], [1, 0], [0, 1], [2, 0], [0, 2]]),
        momentum_matrices=momentum_matrices,
        fourier_vectors=np.stack([vector for vector, _ in terms]),
        fourier_matrices=np.stack([matrix for _, matrix in terms]),
        basis_labels=("bottom", "top"),
        name="two-shell MoTe2 implementation reference",
        energy_unit="meV",
        length_unit="dimensionless",
    )


def test_zero_second_shell_reproduces_builtin_first_shell() -> None:
    positions = jnp.asarray([[0.13, 0.27], [0.41, -0.08]])
    spin_signs = jnp.asarray([1.0, -1.0])
    v1, phi1, omega1 = 0.7, 0.31, -0.4
    actual = _moire_potential_matrix(
        positions,
        jnp.asarray(LATTICE),
        spin_signs,
        v1_mev=v1,
        phi1_rad=phi1,
        omega1_mev=omega1,
        v2_mev=0.0,
        omega2_mev=0.0,
    )

    first_shell = _moire_first_shell_vectors(jnp.asarray(LATTICE))
    projection = positions @ first_shell[0:5:2].T
    v_bottom = -2.0 * v1 * jnp.sum(jnp.cos(projection + phi1), axis=-1)
    v_top = -2.0 * v1 * jnp.sum(jnp.cos(projection - phi1), axis=-1)
    tunneling = omega1 * (
        1.0
        + jnp.exp(1j * (positions @ first_shell[1]))
        + jnp.exp(1j * (positions @ first_shell[2]))
    )
    spin_tunneling = tunneling.real + 1j * spin_signs * tunneling.imag
    expected = jnp.stack(
        [
            jnp.stack([v_bottom, spin_tunneling], axis=-1),
            jnp.stack([jnp.conj(spin_tunneling), v_top], axis=-1),
        ],
        axis=-2,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_k_valley_bands_match_portable_six_parameter_model() -> None:
    reciprocal = PORTABLE_RECIPROCAL_PER_ANGSTROM
    direct_lattice_angstrom = 2.0 * np.pi * np.linalg.inv(reciprocal).T
    source_g = _q_grid(reciprocal, cutoff=2)
    q1 = PORTABLE_KAPPAS_PER_ANGSTROM[0] + source_g
    q2 = PORTABLE_KAPPAS_PER_ANGSTROM[1] + source_g
    built_in_q = -source_g
    parameters_mev = 1000.0 * PORTABLE_PARAMETERS_EV
    a_h, v_cos, v_sin, omega1, v2, omega2 = parameters_mev
    v1 = float(np.hypot(v_cos, v_sin))
    phi1 = float(np.arctan2(v_sin, v_cos))

    for source_k in (
        np.asarray([0.0, 0.0]),
        np.asarray([0.01, -0.02]),
        np.asarray([-0.015, 0.007]),
    ):
        expected, energy_offset = _portable_hole_hamiltonian(
            source_k, q1, q2, parameters_mev
        )
        actual = _implementation_hamiltonian(
            -source_k,
            built_in_q,
            lattice=direct_lattice_angstrom,
            spin_sign=1,
            kinetic_coefficient_mev=a_h,
            v1_mev=v1,
            phi1_rad=phi1,
            omega1_mev=omega1,
            v2_mev=v2,
            omega2_mev=omega2,
        )
        np.testing.assert_allclose(
            np.linalg.eigvalsh(expected),
            energy_offset + np.linalg.eigvalsh(actual),
            rtol=2e-6,
            atol=2e-5,
        )


def test_k_prime_is_time_reversal_of_k_valley() -> None:
    reciprocal = _reciprocal_vectors(LATTICE)
    q_vectors = _q_grid(reciprocal, cutoff=1)
    momentum = np.asarray([0.17, -0.22])
    parameters = {
        "lattice": LATTICE,
        "kinetic_coefficient_mev": 0.43,
        "v1_mev": 0.71,
        "phi1_rad": 0.37,
        "omega1_mev": -0.52,
        "v2_mev": 0.19,
        "omega2_mev": 0.23,
    }
    k_prime = _implementation_hamiltonian(
        momentum, q_vectors, spin_sign=-1, **parameters
    )
    k_at_minus_momentum = _implementation_hamiltonian(
        -momentum, q_vectors, spin_sign=1, **parameters
    )

    indices = np.asarray([(i, j) for i in range(-1, 2) for j in range(-1, 2)])
    inverse_q = np.asarray(
        [np.flatnonzero(np.all(indices == -index, axis=1))[0] for index in indices]
    )
    basis_permutation = np.concatenate([inverse_q, len(q_vectors) + inverse_q])
    expected = k_at_minus_momentum[np.ix_(basis_permutation, basis_permutation)].conj()
    np.testing.assert_allclose(k_prime, expected, rtol=2e-6, atol=2e-5)


def test_noninteracting_many_body_energy_matches_plane_wave_reference() -> None:
    parameters = {
        "lattice": LATTICE,
        "kinetic_coefficient_mev": 0.43,
        "v1_mev": 0.71,
        "phi1_rad": 0.37,
        "omega1_mev": -0.52,
        "v2_mev": 0.19,
        "omega2_mev": 0.23,
    }
    model = _reference_model_from_implementation(**parameters)
    reference = free_fermion_reference(
        model,
        nelec=3,
        primitive_lattice=LATTICE,
        supercell_matrix=np.asarray([[2, 0], [0, 1]]),
        twist=np.asarray([0.13, -0.07]),
        q_index_cutoff=2,
    )

    pooled = []
    for momentum in reference.kpoints:
        matrix = _implementation_hamiltonian(
            momentum,
            reference.q_vectors,
            spin_sign=1,
            **parameters,
        )
        pooled.extend(np.linalg.eigvalsh(matrix))
    expected = float(np.sum(np.sort(pooled)[:3]))
    np.testing.assert_allclose(reference.e_exact, expected, rtol=2e-6, atol=2e-5)
