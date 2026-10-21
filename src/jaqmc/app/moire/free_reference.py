# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Exact non-interacting reference energies for matrix continuum models.

With the Coulomb estimator disabled (``estimators.enabled.coulomb=false``) the
many-body Hamiltonian is a sum of identical one-body operators,

.. math::

    \hat H_0 = \sum_i h(\mathbf r_i),

so its exact ground state is a Slater determinant of the lowest single-particle
states and the benchmark target is the sum of occupied one-body eigenvalues on
the *same* finite simulation-cell grid used by the VMC calculation:

.. math::

    E_{\rm exact}(U=0) = \sum_{a=1}^{N_e} \epsilon_a.

This module diagonalizes :meth:`MatrixContinuumModel.plane_wave_hamiltonian`
at every momentum folded into the simulation supercell and pools the
eigenvalues. Every convention (unit conversion, internal basis, twist,
constant energy offset) is inherited from the same model archive consumed by
the training workflow, which keeps this reference independent of the neural
network while identical in its single-particle physics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import Any

import numpy as np

from jaqmc.app.moire.model import MatrixContinuumModel

__all__ = [
    "FreeFermionReference",
    "check_fourier_commensurateness",
    "folded_k_points",
    "free_fermion_reference",
    "q_cutoff_convergence",
    "q_shell",
]

# Default unit-norm primitive lattice, matching MoireConfig (120-degree rows).
DEFAULT_LATTICE = np.asarray([[np.sqrt(3.0) / 2.0, -0.5], [0.0, 1.0]], dtype=float)


def reciprocal_vectors(lattice: np.ndarray) -> np.ndarray:
    """Return reciprocal lattice rows ``b_i`` with ``b_i . a_j = 2 pi delta``."""
    lattice = np.asarray(lattice, dtype=float)
    return 2.0 * np.pi * np.linalg.inv(lattice).T


def folded_k_points(
    primitive_lattice: np.ndarray,
    supercell_matrix: np.ndarray,
    twist: np.ndarray,
) -> np.ndarray:
    """Return the Cartesian k points folded into the simulation supercell.

    The supercell reciprocal rows are ``B_sim = M^{-T} B``, and the
    ``|det M|`` folded offsets are ``M^{-T} n (mod 1)`` in fractional
    primitive-reciprocal coordinates, for integer ``n``. The twist is given in
    simulation-cell reciprocal fractional coordinates, matching
    ``MoireWavefunction._apply_twist``.

    Args:
        primitive_lattice: Direct primitive lattice rows in dimensionless units.
        supercell_matrix: Integer supercell matrix ``M`` (rows of ``R = M a``).
        twist: Fractional twist with respect to the simulation reciprocal basis.

    Returns:
        Array of shape ``(scale, 2)`` with folded Cartesian k points.

    Raises:
        ValueError: If the folded-point enumeration does not yield
            ``|det M|`` distinct offsets.
    """
    lattice = np.asarray(primitive_lattice, dtype=float)
    supercell = np.asarray(supercell_matrix, dtype=float)
    twist = np.asarray(twist, dtype=float)
    scale = round(abs(np.linalg.det(supercell)))
    reciprocal = reciprocal_vectors(lattice)
    # Rows of the simulation lattice are ``R = M a``; hence ``B_sim = M^{-T} B``,
    # matching MoireConfig.simulation_reciprocal_lattice.
    simulation_reciprocal = np.linalg.inv(supercell).T @ reciprocal
    k_twist = twist @ simulation_reciprocal

    # Enumerate cosets Z^2 / M^T Z^2. A generous integer window guarantees
    # coverage for non-diagonal supercell matrices.
    bound = int(2 * np.max(np.abs(supercell)) + 3)
    minv_t = np.linalg.inv(supercell).T
    fractions: list[np.ndarray] = []
    for n_x in range(-bound, bound + 1):
        for n_y in range(-bound, bound + 1):
            fraction = (minv_t @ np.asarray([n_x, n_y], dtype=float)) % 1.0
            fraction = np.where(np.isclose(fraction, 1.0, atol=1e-9), 0.0, fraction)
            if not any(np.allclose(fraction, other, atol=1e-9) for other in fractions):
                fractions.append(fraction)
    if len(fractions) != scale:
        raise ValueError(
            f"Expected {scale} folded k points from supercell matrix, found "
            f"{len(fractions)}; check that the matrix is integer and non-singular."
        )
    return k_twist[None, :] + np.asarray(fractions) @ reciprocal


def q_shell(primitive_lattice: np.ndarray, index_cutoff: int) -> np.ndarray:
    """Return symmetric moire reciprocal-plane-wave set ``Q = i b1 + j b2``.

    Args:
        primitive_lattice: Direct primitive lattice rows in dimensionless units.
        index_cutoff: Include all ``(i, j)`` with ``|i|, |j| <= cutoff``.

    Returns:
        Q vectors sorted by norm with shape ``(n_q, 2)``.

    Raises:
        ValueError: If ``index_cutoff`` is negative.
    """
    if index_cutoff < 0:
        raise ValueError("index_cutoff must be non-negative.")
    reciprocal = reciprocal_vectors(np.asarray(primitive_lattice, dtype=float))
    vectors = [
        i * reciprocal[0] + j * reciprocal[1]
        for i in range(-index_cutoff, index_cutoff + 1)
        for j in range(-index_cutoff, index_cutoff + 1)
    ]
    q_vectors = np.asarray(vectors, dtype=float)
    order = np.argsort(np.linalg.norm(q_vectors, axis=1))
    return q_vectors[order]


def check_fourier_commensurateness(
    model: MatrixContinuumModel,
    primitive_lattice: np.ndarray,
    supercell_matrix: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Check that model Fourier vectors tile the simulation supercell.

    A Fourier harmonic ``g`` is periodic on the simulation cell ``R = M a``
    only when ``g . R_i / (2 pi)`` is integer. Non-commensurate vectors would
    be dropped silently by the plane-wave Q matching, producing a reference
    with missing couplings; this guard turns that into a loud failure.

    Args:
        model: Dimensionless model used by the VMC workflow.
        primitive_lattice: Direct primitive lattice rows.
        supercell_matrix: Integer supercell matrix.
        tolerance: Acceptance tolerance on the integer phase conditions.

    Returns:
        Integer coefficients ``(n_terms, 2, 2)`` giving each stored Fourier
        vector (axis 1) in the reciprocal basis and supercell phase indices.

    Raises:
        ValueError: If a Fourier vector is not commensurate with the cell.
    """
    reciprocal = reciprocal_vectors(np.asarray(primitive_lattice, dtype=float))
    coefficients = model.fourier_vectors @ np.linalg.inv(reciprocal)
    supercell_phases = coefficients @ np.asarray(supercell_matrix, dtype=float).T
    if np.any(np.abs(supercell_phases - np.round(supercell_phases)) > tolerance):
        offenders = model.fourier_vectors[
            np.any(
                np.abs(supercell_phases - np.round(supercell_phases)) > tolerance,
                axis=1,
            )
        ]
        raise ValueError(
            "Model Fourier vectors are not commensurate with the simulation "
            "supercell; the plane-wave reference would silently drop "
            f"couplings. Offending vectors (dimensionless): {offenders.tolist()}. "
            "Align system.lattice_vectors / moire geometry with the model's "
            "Fourier lattice before running the free-fermion benchmark."
        )
    return np.stack([coefficients, supercell_phases], axis=1)


@dataclass(frozen=True)
class OccupiedState:
    """One occupied single-particle state in the pooled supercell spectrum."""

    k_index: int
    band_index: int
    energy: float


@dataclass(frozen=True)
class FreeFermionReference:
    """Exact non-interacting result for one finite supercell and Q cutoff.

    Attributes:
        e_exact: Sum of the ``nelec`` lowest pooled eigenvalues (meV).
        kpoints: Folded Cartesian k points with shape ``(scale, 2)``.
        q_vectors: Plane-wave vectors used in the diagonalization.
        eigenvalues_per_k: Sorted eigenvalues for each folded k point.
        occupied: Selected occupied states in eigenvalue order.
        homo: Highest occupied eigenvalue.
        lumo: Lowest unoccupied eigenvalue, or ``None`` at band-space cutoff.
    """

    e_exact: float
    kpoints: np.ndarray
    q_vectors: np.ndarray
    eigenvalues_per_k: list[np.ndarray]
    occupied: list[OccupiedState]
    homo: float
    lumo: float | None

    @property
    def gap(self) -> float | None:
        """HOMO-LUMO gap at the current Q cutoff, or ``None`` if exhausted."""
        return None if self.lumo is None else self.lumo - self.homo


def _restricted_indices(
    n_q: int, n_internal: int, internal_indices: np.ndarray | None
) -> np.ndarray:
    if internal_indices is None:
        return np.arange(n_q * n_internal)
    internal_indices = np.asarray(internal_indices, dtype=int)
    if np.any(internal_indices < 0) or np.any(internal_indices >= n_internal):
        raise ValueError("internal_indices outside the model internal basis.")
    return np.asarray(
        [q * n_internal + a for q in range(n_q) for a in internal_indices]
    )


def free_fermion_reference(
    model: MatrixContinuumModel,
    nelec: int,
    *,
    primitive_lattice: np.ndarray = DEFAULT_LATTICE,
    supercell_matrix: np.ndarray = np.eye(2, dtype=int),
    twist: np.ndarray = np.zeros(2),
    q_index_cutoff: int = 2,
    internal_indices: np.ndarray | None = None,
    matching_tolerance: float = 1e-8,
    commensurateness_tolerance: float = 1e-8,
) -> FreeFermionReference:
    """Compute ``E_exact = sum of the nelec lowest eigenvalues``.

    Args:
        model: Dimensionless model as produced by ``to_dimensionless``.
        nelec: Number of particles in the fixed physical-spin sector.
        primitive_lattice: Primitive lattice rows (dimensionless, unit norm).
        supercell_matrix: Integer simulation-cell matrix.
        twist: Twist in simulation-cell reciprocal fractional coordinates.
        q_index_cutoff: Moire plane-wave index cutoff (``|i|, |j|``).
        internal_indices: Optional internal-basis indices restricting the
            one-body Hilbert space to a conserved sector (e.g. internal spin).
        matching_tolerance: Forwarded to ``plane_wave_hamiltonian``.
        commensurateness_tolerance: Geometry alignment tolerance.

    Returns:
        The pooled, sorted reference spectrum and occupied-state bookkeeping.

    Raises:
        ValueError: If the particle count, internal indices, geometry, or Q
            cutoff is invalid or incommensurate with the model.
    """
    supercell = np.asarray(supercell_matrix, dtype=int)
    if nelec <= 0:
        raise ValueError("nelec must be positive.")
    check_fourier_commensurateness(
        model,
        primitive_lattice,
        supercell,
        tolerance=commensurateness_tolerance,
    )
    kpoints = folded_k_points(primitive_lattice, supercell, np.asarray(twist))
    q_vectors = q_shell(primitive_lattice, q_index_cutoff)

    n_internal = model.internal_dim
    subspace = _restricted_indices(len(q_vectors), n_internal, internal_indices)
    eigenvalues_per_k: list[np.ndarray] = []
    pooled: list[tuple[float, int, int]] = []
    for k_index, kpoint in enumerate(kpoints):
        hamiltonian = model.plane_wave_hamiltonian(
            np.asarray(kpoint, dtype=float),
            q_vectors,
            matching_tolerance=matching_tolerance,
        )
        hamiltonian = hamiltonian[np.ix_(subspace, subspace)]
        hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
        eigenvalues = np.linalg.eigvalsh(hamiltonian)
        if not np.all(np.isfinite(eigenvalues)):
            raise ValueError(f"Non-finite eigenvalues at k point {kpoint}.")
        eigenvalues_per_k.append(eigenvalues)
        pooled.extend(
            (float(energy), k_index, band) for band, energy in enumerate(eigenvalues)
        )

    if nelec > len(pooled):
        raise ValueError(
            f"nelec={nelec} exceeds {len(pooled)} available one-body states at "
            f"Q index cutoff {q_index_cutoff}; increase q_index_cutoff."
        )
    pooled.sort(key=itemgetter(0))
    occupied = [
        OccupiedState(k_index, band, energy) for energy, k_index, band in pooled[:nelec]
    ]
    lumo = float(pooled[nelec][0]) if nelec < len(pooled) else None
    return FreeFermionReference(
        e_exact=float(sum(state.energy for state in occupied)),
        kpoints=kpoints,
        q_vectors=q_vectors,
        eigenvalues_per_k=eigenvalues_per_k,
        occupied=occupied,
        homo=occupied[-1].energy,
        lumo=lumo,
    )


def q_cutoff_convergence(
    model: MatrixContinuumModel,
    nelec: int,
    cutoffs: list[int],
    **reference_kwargs: Any,
) -> list[dict[str, float | int | None]]:
    """Evaluate the reference energy at increasing Q cutoffs.

    The one-body energy is converged when successive changes fall below the
    target NQS accuracy; the VMC potential uses the continuous Fourier series,
    so the finite Q expansion is the only approximation on the reference side.

    Returns:
        One dictionary per cutoff with the energy, increment, and gap.
    """
    results: list[dict[str, float | int | None]] = []
    previous: float | None = None
    for cutoff in cutoffs:
        reference = free_fermion_reference(
            model, nelec, q_index_cutoff=cutoff, **reference_kwargs
        )
        results.append(
            {
                "q_index_cutoff": cutoff,
                "n_q": len(reference.q_vectors),
                "e_exact": reference.e_exact,
                "delta_meV": None if previous is None else reference.e_exact - previous,
                "gap_meV": reference.gap,
            }
        )
        previous = reference.e_exact
    return results


def _build_model(
    args: argparse.Namespace,
) -> tuple[MatrixContinuumModel, dict[str, Any]]:
    model = MatrixContinuumModel.load(args.model)
    if args.length_scale_nm is not None:
        model = model.to_dimensionless(args.length_scale_nm)
    elif model.length_unit.lower() != "dimensionless":
        raise ValueError(
            "--length-scale-nm is required for models with physical length units."
        )
    geometry = {
        "primitive_lattice": np.asarray(args.lattice, dtype=float),
        "supercell_matrix": np.asarray(args.supercell, dtype=int),
        "twist": np.asarray(args.twist, dtype=float),
    }
    return model, geometry


def main(argv: list[str] | None = None) -> None:
    """Command-line entry: print the free-fermion reference as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--length-scale-nm", type=float, default=None)
    parser.add_argument("--nelec", required=True, type=int)
    parser.add_argument("--lattice", type=json.loads, default=DEFAULT_LATTICE.tolist())
    parser.add_argument("--supercell", type=json.loads, default=[[1, 0], [0, 1]])
    parser.add_argument("--twist", type=json.loads, default=[0.0, 0.0])
    parser.add_argument("--q-index-cutoff", type=int, default=2)
    parser.add_argument(
        "--convergence-cutoffs",
        type=lambda value: [int(item) for item in value.split(",") if item],
        default=None,
    )
    parser.add_argument("--internal-indices", type=str, default=None)
    args = parser.parse_args(argv)

    model, geometry = _build_model(args)
    internal = (
        None
        if args.internal_indices is None
        else np.asarray([int(item) for item in args.internal_indices.split(",")])
    )
    reference = free_fermion_reference(
        model,
        args.nelec,
        internal_indices=internal,
        q_index_cutoff=args.q_index_cutoff,
        **geometry,
    )
    payload: dict[str, Any] = {
        "model": model.name,
        "energy_unit": model.energy_unit,
        "nelec": args.nelec,
        "n_k_points": len(reference.kpoints),
        "n_q": len(reference.q_vectors),
        "e_exact": reference.e_exact,
        "homo": reference.homo,
        "lumo": reference.lumo,
        "gap": reference.gap,
        "occupied": [
            {"k": state.k_index, "band": state.band_index, "energy": state.energy}
            for state in reference.occupied
        ],
    }
    if args.convergence_cutoffs:
        payload["q_convergence"] = q_cutoff_convergence(
            model,
            args.nelec,
            args.convergence_cutoffs,
            internal_indices=internal,
            **geometry,
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
