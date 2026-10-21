# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for the exact non-interacting (U=0) reference energy module."""

import contextlib
import io
import json

import numpy as np
import pytest

from jaqmc.app.moire.free_reference import (
    DEFAULT_LATTICE,
    folded_k_points,
    free_fermion_reference,
    main,
    q_cutoff_convergence,
    q_shell,
    reciprocal_vectors,
)
from jaqmc.app.moire.model import MatrixContinuumModel

RECIPROCAL = reciprocal_vectors(DEFAULT_LATTICE)
B_NORM = 4.0 * np.pi / np.sqrt(3.0)


def free_particle_model(
    *,
    alpha: float = 0.5,
    constant_diagonal: np.ndarray | None = None,
    fourier_vectors: np.ndarray | None = None,
    fourier_matrices: np.ndarray | None = None,
    energy_unit: str = "meV",
    length_unit: str = "dimensionless",
) -> MatrixContinuumModel:
    """Build h(p) = alpha p^2 I + C, optionally with plane-wave couplings."""
    constant_diagonal = (
        np.zeros(1) if constant_diagonal is None else np.asarray(constant_diagonal)
    )
    n_internal = len(constant_diagonal)
    identity = np.eye(n_internal, dtype=complex)
    powers = [(2, 0), (0, 2), (0, 0)]
    matrices = np.asarray(
        [alpha * identity, alpha * identity, np.diag(constant_diagonal).astype(complex)]
    )
    vectors = np.zeros((0, 2)) if fourier_vectors is None else fourier_vectors
    couplings = (
        np.zeros((0, n_internal, n_internal), dtype=complex)
        if fourier_matrices is None
        else fourier_matrices
    )
    return MatrixContinuumModel(
        momentum_powers=np.asarray(powers),
        momentum_matrices=matrices,
        fourier_vectors=vectors,
        fourier_matrices=couplings,
        energy_unit=energy_unit,
        length_unit=length_unit,
        name="analytic free test model",
    )


def fractional_coordinates(kpoints: np.ndarray) -> np.ndarray:
    """Express Cartesian k points in fractional primitive-reciprocal coords."""
    return kpoints @ np.linalg.inv(RECIPROCAL)


def test_folded_k_points_primitive_and_supercell() -> None:
    def fractions_of(supercell, twist=(0.0, 0.0)):
        points = folded_k_points(DEFAULT_LATTICE, supercell, twist)
        fractions = fractional_coordinates(points)
        # Wrap into [0, 1), tolerating ~1e-16 floating wrap-around.
        fractions = np.mod(fractions, 1.0)
        fractions[np.isclose(fractions, 1.0, atol=1e-8)] = 0.0
        fractions = np.round(fractions, 9)
        return {tuple(row) for row in fractions}

    assert fractions_of(np.eye(2, dtype=int)) == {(0.0, 0.0)}
    assert fractions_of([[1, 0], [0, 2]]) == {(0.0, 0.0), (0.0, 0.5)}
    assert fractions_of([[2, 0], [0, 2]]) == {
        (0.0, 0.0),
        (0.5, 0.0),
        (0.0, 0.5),
        (0.5, 0.5),
    }


def test_folded_k_points_with_twist() -> None:
    # Twist is fractional in the *simulation* reciprocal basis, whose rows for
    # M = diag(1, 2) are (b1, b2/2).
    kpoints = folded_k_points(DEFAULT_LATTICE, [[1, 0], [0, 2]], [0.25, 0.25])
    k_twist = 0.25 * RECIPROCAL[0] + 0.125 * RECIPROCAL[1]
    expected = [k_twist, k_twist + 0.5 * RECIPROCAL[1]]
    assert len(kpoints) == 2
    for point in expected:
        assert any(np.allclose(point, other) for other in kpoints)


def test_q_shell_symmetric_and_sorted() -> None:
    # Index cutoff |i|, |j| <= 1 yields a 3x3 set of nine reciprocal vectors.
    shell = q_shell(DEFAULT_LATTICE, 1)
    assert len(shell) == 9
    # Inversion symmetry is required for a real finite-system spectrum.
    for vector in shell:
        assert any(np.allclose(vector, -other) for other in shell)
    norms = np.linalg.norm(shell, axis=1)
    assert np.all(np.diff(norms) >= -1e-12)
    assert len(q_shell(DEFAULT_LATTICE, 2)) == 25


def test_free_particle_primitive_shell_filling() -> None:
    model = free_particle_model()
    reference = free_fermion_reference(
        model, nelec=1, supercell_matrix=np.eye(2, dtype=int), q_index_cutoff=1
    )
    assert reference.e_exact == pytest.approx(0.0, abs=1e-12)
    # Six moire reciprocal vectors form the first shell around Gamma.
    assert reference.gap == pytest.approx(0.5 * B_NORM**2)
    assert reference.lumo == pytest.approx(8.0 * np.pi**2 / 3.0)

    reference = free_fermion_reference(
        model, nelec=7, supercell_matrix=np.eye(2, dtype=int), q_index_cutoff=1
    )
    assert reference.e_exact == pytest.approx(6.0 * 8.0 * np.pi**2 / 3.0)


def test_constant_matrix_shifts_total_energy() -> None:
    model = free_particle_model(constant_diagonal=[0.3])
    reference = free_fermion_reference(
        model, nelec=1, supercell_matrix=np.eye(2, dtype=int), q_index_cutoff=1
    )
    assert reference.e_exact == pytest.approx(0.3)
    assert reference.gap == pytest.approx(8.0 * np.pi**2 / 3.0)

    # Filling one of the sixfold-degenerate first shell leaves gap zero but
    # shifts every occupied level by the same constant offset.
    filled_shell = free_fermion_reference(
        model, nelec=2, supercell_matrix=np.eye(2, dtype=int), q_index_cutoff=1
    )
    assert filled_shell.e_exact == pytest.approx(0.6 + 8.0 * np.pi**2 / 3.0)
    assert filled_shell.gap == pytest.approx(0.0, abs=1e-10)


def test_supercell_folding_sums_reduced_momenta() -> None:
    model = free_particle_model()
    reference = free_fermion_reference(
        model,
        nelec=2,
        supercell_matrix=[[1, 0], [0, 2]],
        twist=[0.0, 0.0],
        q_index_cutoff=1,
    )
    # k = 0 state plus the reduced k = b2/2 state, epsilon = alpha |b2/2|^2.
    assert reference.e_exact == pytest.approx(2.0 * np.pi**2 / 3.0)
    # k2 = b2/2 has a degenerate partner via Q = -b2, so the pooled gap closes.
    assert reference.gap == pytest.approx(0.0, abs=1e-10)


def test_internal_sector_restriction() -> None:
    model = free_particle_model(constant_diagonal=[0.1, 0.7])
    full = free_fermion_reference(
        model, nelec=1, supercell_matrix=np.eye(2, dtype=int), q_index_cutoff=0
    )
    assert full.e_exact == pytest.approx(0.1)
    assert full.gap == pytest.approx(0.6)

    spin_sector = free_fermion_reference(
        model,
        nelec=1,
        supercell_matrix=np.eye(2, dtype=int),
        q_index_cutoff=0,
        internal_indices=np.asarray([1]),
    )
    assert spin_sector.e_exact == pytest.approx(0.7)

    degenerate_model = free_particle_model(constant_diagonal=[0.1, 0.1])
    two_particles = free_fermion_reference(
        degenerate_model,
        nelec=2,
        supercell_matrix=np.eye(2, dtype=int),
        q_index_cutoff=1,
    )
    # Both internal components host a zero-momentum state at c = 0.1.
    assert two_particles.e_exact == pytest.approx(0.2)


def test_fourier_coupling_lowers_energy_and_q_converges() -> None:
    g = RECIPROCAL[0]
    model = free_particle_model(
        fourier_vectors=np.asarray([g]),
        fourier_matrices=np.asarray([[[0.3 + 0.0j]]]),
    )
    uncoupled = free_fermion_reference(
        model, nelec=1, supercell_matrix=np.eye(2, dtype=int), q_index_cutoff=0
    )
    coupled = free_fermion_reference(
        model, nelec=1, supercell_matrix=np.eye(2, dtype=int), q_index_cutoff=1
    )
    assert uncoupled.e_exact == pytest.approx(0.0, abs=1e-12)
    # A real coupling opens bonding/antibonding combinations: lowest < 0.
    assert coupled.e_exact < -1e-8

    table = q_cutoff_convergence(
        model,
        nelec=1,
        cutoffs=[0, 1, 2],
        supercell_matrix=np.eye(2, dtype=int),
    )
    energies = [float(row["e_exact"]) for row in table]
    assert energies[0] == pytest.approx(0.0, abs=1e-12)
    assert energies[1] < energies[0]
    # Variational monotonicity in the growing Q subspace; increments diminish.
    assert energies[2] <= energies[1]
    assert abs(energies[2] - energies[1]) < abs(energies[1] - energies[0])


def test_noncommensurate_fourier_vector_is_rejected() -> None:
    angle = np.deg2rad(30.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    bad_vector = rotation @ RECIPROCAL[0]
    model = free_particle_model(
        fourier_vectors=np.asarray([bad_vector]),
        fourier_matrices=np.asarray([[[1.0 + 0.0j]]]),
    )
    with pytest.raises(ValueError, match="commensurate"):
        free_fermion_reference(
            model, nelec=1, supercell_matrix=np.eye(2, dtype=int), q_index_cutoff=1
        )


def test_nelec_beyond_q_space_raises() -> None:
    model = free_particle_model()
    with pytest.raises(ValueError, match="exceeds"):
        free_fermion_reference(
            model, nelec=99, supercell_matrix=np.eye(2, dtype=int), q_index_cutoff=0
        )


def test_cli_dimensionless_and_physical_units(tmp_path) -> None:
    dimensionless = free_particle_model(constant_diagonal=[0.3])
    model_path = tmp_path / "dimensionless.npz"
    dimensionless.save(model_path)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        main(
            [
                "--model",
                str(model_path),
                "--nelec",
                "1",
                "--supercell",
                "[[1,0],[0,1]]",
                "--q-index-cutoff",
                "0",
            ]
        )
    payload = json.loads(stdout.getvalue())
    assert payload["e_exact"] == pytest.approx(0.3)
    assert payload["n_k_points"] == 1 and payload["n_q"] == 1

    # alpha p^2 with alpha = 0.2 eV/A^2 and length scale 2 nm maps to
    # alpha' = 0.2 * 1000 / 20^2 = 0.5 meV in dimensionless coordinates;
    # c = 0.0003 eV -> 0.3 meV.
    physical = free_particle_model(
        constant_diagonal=[0.0003],
        alpha=0.2,
        energy_unit="eV",
        length_unit="angstrom",
    )
    physical_path = tmp_path / "physical.npz"
    physical.save(physical_path)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        main(
            [
                "--model",
                str(physical_path),
                "--length-scale-nm",
                "2.0",
                "--nelec",
                "7",
                "--supercell",
                "[[1,0],[0,1]]",
                "--q-index-cutoff",
                "1",
            ]
        )
    payload = json.loads(stdout.getvalue())
    assert payload["e_exact"] == pytest.approx(
        7.0 * 0.3 + 6.0 * 8.0 * np.pi**2 / 3.0, rel=1e-10
    )
