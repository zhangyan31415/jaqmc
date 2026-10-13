# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Material-independent matrix continuum models for moire systems."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from upath import UPath

from jaqmc.utils.units import ONE_HARTREE_IN_MEV, ONE_NM_IN_BOHR

__all__ = ["MatrixContinuumModel"]

_ENERGY_TO_MEV = {
    "mev": 1.0,
    "ev": 1000.0,
    "hartree": ONE_HARTREE_IN_MEV,
}


def _validate_coordinate_transform(transform: np.ndarray | None) -> np.ndarray:
    result = np.eye(2) if transform is None else np.asarray(transform, dtype=float)
    if result.shape != (2, 2) or not np.allclose(
        result.T @ result,
        np.eye(2),
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ValueError(
            "source_from_model_coordinates must be a real orthogonal "
            "matrix with shape (2, 2)."
        )
    return result


def _validate_momentum_terms(
    powers: np.ndarray, matrices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if powers.ndim != 2 or powers.shape[1:] != (2,):
        raise ValueError("momentum_powers must have shape (n_terms, 2).")
    if not np.issubdtype(powers.dtype, np.integer) or np.any(powers < 0):
        raise ValueError("momentum_powers must contain non-negative integers.")
    if len({tuple(power) for power in powers.tolist()}) != len(powers):
        raise ValueError("momentum_powers must not contain duplicates.")
    if matrices.ndim != 3:
        raise ValueError(
            "momentum_matrices must have shape (n_terms, n_internal, n_internal)."
        )
    if matrices.shape[0] != len(powers):
        raise ValueError("Momentum powers and matrices must have equal lengths.")
    if matrices.shape[1] != matrices.shape[2]:
        raise ValueError("Momentum coefficient matrices must be square.")
    if matrices.shape[1] <= 0:
        raise ValueError("The internal basis must be non-empty.")
    if not np.all(np.isfinite(matrices)):
        raise ValueError("Momentum matrices must contain only finite values.")
    if not np.allclose(
        matrices,
        matrices.conj().transpose(0, 2, 1),
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ValueError("Every momentum coefficient matrix must be Hermitian.")
    return powers.astype(np.int64), matrices


def _validate_fourier_vectors(vectors: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(vectors)):
        raise ValueError("Fourier vectors must contain only finite values.")
    for index, vector in enumerate(vectors):
        if np.linalg.norm(vector) <= 1e-12:
            raise ValueError(
                "Zero-vector Fourier terms belong in momentum power (0, 0)."
            )
        for previous in vectors[:index]:
            if np.allclose(vector, previous, atol=1e-10, rtol=1e-10):
                raise ValueError("fourier_vectors must not contain duplicates.")
            if np.allclose(vector, -previous, atol=1e-10, rtol=1e-10):
                raise ValueError("Store only one vector from each +/-g Fourier pair.")
    return vectors.astype(float)


def _validate_fourier_terms(
    vectors: np.ndarray, matrices: np.ndarray, n_internal: int
) -> tuple[np.ndarray, np.ndarray]:
    if vectors.ndim != 2 or vectors.shape[1:] != (2,):
        raise ValueError("fourier_vectors must have shape (n_terms, 2).")
    if matrices.shape != (len(vectors), n_internal, n_internal):
        raise ValueError(
            "fourier_matrices must have shape "
            "(n_fourier_terms, n_internal, n_internal)."
        )
    if not np.all(np.isfinite(matrices)):
        raise ValueError("Fourier matrices must contain only finite values.")
    return _validate_fourier_vectors(vectors), matrices


@dataclass(frozen=True)
class MatrixContinuumModel:
    r"""A finite-dimensional continuum Hamiltonian specification.

    The translation-invariant part is represented as a polynomial in Cartesian
    momentum,

    .. math::

        h_0(\mathbf p) = \sum_t p_x^{n_{t,x}}p_y^{n_{t,y}} A_t.

    Each stored Fourier pair contributes

    .. math::

        V_g(\mathbf r) = B_g e^{i\mathbf g\cdot\mathbf r}
            + B_g^\dagger e^{-i\mathbf g\cdot\mathbf r}.

    This one-sided representation guarantees a Hermitian real-space potential
    without requiring both ``g`` and ``-g`` in the archive. Plane-wave matrices
    use the convention :math:`e^{i(\mathbf k+\mathbf Q)\cdot\mathbf r}`.

    Args:
        momentum_powers: Non-negative integer powers ``(nx, ny)`` with shape
            ``(n_momentum_terms, 2)``.
        momentum_matrices: Hermitian coefficient matrices with shape
            ``(n_momentum_terms, n_internal, n_internal)``.
        fourier_vectors: One vector from each nonzero ``+/-g`` pair, shaped
            ``(n_fourier_terms, 2)``.
        fourier_matrices: Coefficients ``B_g`` with shape
            ``(n_fourier_terms, n_internal, n_internal)``.
        basis_labels: Optional labels for the internal basis.
        name: Human-readable model name.
        energy_unit: Energy unit shared by all coefficient matrices.
        length_unit: Length unit reciprocal to the momentum and Fourier vectors.
    """

    momentum_powers: np.ndarray
    momentum_matrices: np.ndarray
    fourier_vectors: np.ndarray
    fourier_matrices: np.ndarray
    basis_labels: tuple[str, ...] = ()
    name: str = ""
    energy_unit: str = "meV"
    length_unit: str = "dimensionless"

    def __post_init__(self) -> None:
        powers, momentum_matrices = _validate_momentum_terms(
            np.asarray(self.momentum_powers),
            np.asarray(self.momentum_matrices),
        )
        fourier_vectors, fourier_matrices = _validate_fourier_terms(
            np.asarray(self.fourier_vectors),
            np.asarray(self.fourier_matrices),
            momentum_matrices.shape[1],
        )
        n_internal = momentum_matrices.shape[1]
        if self.basis_labels and len(self.basis_labels) != n_internal:
            raise ValueError(
                "basis_labels must be empty or match the internal dimension."
            )
        if not self.energy_unit or not self.length_unit:
            raise ValueError("energy_unit and length_unit must be non-empty.")

        object.__setattr__(self, "momentum_powers", powers)
        object.__setattr__(self, "momentum_matrices", momentum_matrices)
        object.__setattr__(self, "fourier_vectors", fourier_vectors)
        object.__setattr__(self, "fourier_matrices", fourier_matrices)

    @property
    def internal_dim(self) -> int:
        """Number of layer/orbital/spin components in the internal basis."""
        return self.momentum_matrices.shape[1]

    @property
    def constant_matrix(self) -> np.ndarray:
        """Return the momentum-independent internal matrix."""
        matches = np.all(self.momentum_powers == 0, axis=1)
        if not np.any(matches):
            return np.zeros(
                (self.internal_dim, self.internal_dim),
                dtype=self.momentum_matrices.dtype,
            )
        return self.momentum_matrices[np.flatnonzero(matches)[0]]

    @property
    def differential_terms(self) -> tuple[np.ndarray, np.ndarray]:
        """Return nonconstant momentum powers and coefficient matrices."""
        nonconstant = np.any(self.momentum_powers != 0, axis=1)
        return (
            self.momentum_powers[nonconstant],
            self.momentum_matrices[nonconstant],
        )

    @classmethod
    def from_fitted_terms(
        cls,
        *,
        momentum_powers: np.ndarray,
        momentum_matrices: np.ndarray,
        fourier_vectors: np.ndarray,
        fourier_matrices: np.ndarray,
        source_from_model_coordinates: np.ndarray | None = None,
        basis_labels: tuple[str, ...] = (),
        name: str = "",
        energy_unit: str = "meV",
        length_unit: str = "dimensionless",
    ) -> "MatrixContinuumModel":
        r"""Import fitted terms into the canonical JaQMC convention.

        Source momentum components are related to model components by

        .. math::

            \mathbf p_{\rm source} = T\mathbf p_{\rm model}.

        The real orthogonal matrix ``T`` may swap, reflect, or rotate axes.
        Momentum monomials are expanded exactly in model coordinates, while
        Fourier row vectors transform as ``g_model = g_source @ T``.

        Returns:
            Validated canonical matrix continuum model.
        """
        transform = _validate_coordinate_transform(source_from_model_coordinates)

        powers, matrices = _validate_momentum_terms(
            np.asarray(momentum_powers),
            np.asarray(momentum_matrices),
        )
        transformed_terms: dict[tuple[int, int], np.ndarray] = {}
        for (power_x, power_y), matrix in zip(powers, matrices, strict=True):
            for source_x_from_model_x in range(power_x + 1):
                source_x_from_model_y = power_x - source_x_from_model_x
                x_factor = (
                    math.comb(power_x, source_x_from_model_x)
                    * transform[0, 0] ** source_x_from_model_x
                    * transform[0, 1] ** source_x_from_model_y
                )
                for source_y_from_model_x in range(power_y + 1):
                    source_y_from_model_y = power_y - source_y_from_model_x
                    factor = (
                        x_factor
                        * math.comb(power_y, source_y_from_model_x)
                        * transform[1, 0] ** source_y_from_model_x
                        * transform[1, 1] ** source_y_from_model_y
                    )
                    if abs(factor) <= 1e-15:
                        continue
                    model_power = (
                        source_x_from_model_x + source_y_from_model_x,
                        source_x_from_model_y + source_y_from_model_y,
                    )
                    transformed_terms[model_power] = (
                        transformed_terms.get(
                            model_power,
                            np.zeros_like(matrix),
                        )
                        + factor * matrix
                    )

        transformed_powers = np.asarray(sorted(transformed_terms), dtype=np.int64)
        transformed_matrices = np.stack(
            [transformed_terms[tuple(power)] for power in transformed_powers]
        )
        transformed_vectors = np.asarray(fourier_vectors) @ transform
        return cls(
            momentum_powers=transformed_powers,
            momentum_matrices=transformed_matrices,
            fourier_vectors=transformed_vectors,
            fourier_matrices=fourier_matrices,
            basis_labels=basis_labels,
            name=name,
            energy_unit=energy_unit,
            length_unit=length_unit,
        )

    @staticmethod
    def canonical_q_vectors(
        q_vectors: np.ndarray,
        *,
        source_from_model_coordinates: np.ndarray | None = None,
        plane_wave_q_sign: int = 1,
    ) -> np.ndarray:
        r"""Convert source plane-wave Q vectors to the JaQMC convention.

        A source basis
        :math:`e^{i(\mathbf k+s\mathbf Q)\cdot\mathbf r}` maps to canonical
        ``k+Q`` vectors as ``Q_model = s * Q_source @ T``. The physical Fourier
        vectors themselves do not receive the Q sign because they define the
        real-space potential independently of the plane-wave basis labeling.

        Returns:
            Q vectors in model coordinates and the canonical ``k+Q`` convention.

        Raises:
            ValueError: If vectors, coordinate transform, or Q sign are invalid.
        """
        vectors = np.asarray(q_vectors)
        if vectors.ndim != 2 or vectors.shape[1:] != (2,):
            raise ValueError("q_vectors must have shape (nq, 2).")
        transform = _validate_coordinate_transform(source_from_model_coordinates)
        if plane_wave_q_sign not in (-1, 1):
            raise ValueError("plane_wave_q_sign must be either -1 or 1.")
        return plane_wave_q_sign * vectors @ transform

    def to_dimensionless(self, length_scale_nm: float) -> "MatrixContinuumModel":
        """Convert coefficients to meV and a dimensionless spatial coordinate.

        ``length_scale_nm`` defines the physical length represented by one
        dimensionless position unit. Momentum coefficients of total degree
        ``d`` are scaled by ``energy_factor / length_factor**d``; reciprocal
        vectors are scaled by ``length_factor``.

        Returns:
            Equivalent model with ``energy_unit="meV"`` and
            ``length_unit="dimensionless"``.

        Raises:
            ValueError: If units are unsupported or the length scale is invalid.
        """
        if not np.isfinite(length_scale_nm) or length_scale_nm <= 0:
            raise ValueError("length_scale_nm must be positive and finite.")
        try:
            energy_factor = _ENERGY_TO_MEV[self.energy_unit.lower()]
        except KeyError as exc:
            raise ValueError(
                "Unsupported energy_unit. Expected one of "
                f"{sorted(_ENERGY_TO_MEV)}, got {self.energy_unit!r}."
            ) from exc

        length_unit = self.length_unit.lower()
        if length_unit == "dimensionless":
            length_factor = 1.0
        elif length_unit == "angstrom":
            length_factor = 10.0 * length_scale_nm
        elif length_unit == "nm":
            length_factor = length_scale_nm
        elif length_unit == "bohr":
            length_factor = length_scale_nm * ONE_NM_IN_BOHR
        else:
            raise ValueError(
                "Unsupported length_unit. Expected one of "
                "['angstrom', 'bohr', 'dimensionless', 'nm'], "
                f"got {self.length_unit!r}."
            )

        degrees = np.sum(self.momentum_powers, axis=1)
        momentum_scale = energy_factor / length_factor**degrees
        return MatrixContinuumModel(
            momentum_powers=self.momentum_powers,
            momentum_matrices=self.momentum_matrices * momentum_scale[:, None, None],
            fourier_vectors=self.fourier_vectors * length_factor,
            fourier_matrices=self.fourier_matrices * energy_factor,
            basis_labels=self.basis_labels,
            name=self.name,
            energy_unit="meV",
            length_unit="dimensionless",
        )

    def momentum_hamiltonian(self, momentum: np.ndarray) -> np.ndarray:
        """Evaluate the translation-invariant internal Hamiltonian.

        Returns:
            Internal Hamiltonian matrices with the leading shape of ``momentum``.

        Raises:
            ValueError: If the trailing momentum dimension is not two.
        """
        momentum = np.asarray(momentum)
        if momentum.shape[-1:] != (2,):
            raise ValueError("momentum must have trailing shape (2,).")
        monomials = np.prod(
            momentum[..., None, :] ** self.momentum_powers,
            axis=-1,
        )
        return np.einsum(
            "...t,tij->...ij", monomials, self.momentum_matrices, optimize=True
        )

    def periodic_potential(self, position: np.ndarray) -> np.ndarray:
        """Evaluate the Hermitian matrix-valued periodic potential.

        Returns:
            Potential matrices with the leading shape of ``position``.

        Raises:
            ValueError: If the trailing position dimension is not two.
        """
        position = np.asarray(position)
        if position.shape[-1:] != (2,):
            raise ValueError("position must have trailing shape (2,).")
        phase = np.exp(1j * (position @ self.fourier_vectors.T))
        forward = np.einsum(
            "...g,gij->...ij", phase, self.fourier_matrices, optimize=True
        )
        return forward + np.swapaxes(forward.conj(), -1, -2)

    def plane_wave_hamiltonian(
        self,
        momentum: np.ndarray,
        q_vectors: np.ndarray,
        *,
        matching_tolerance: float = 1e-8,
    ) -> np.ndarray:
        r"""Build the plane-wave Hamiltonian in ``Q -> internal`` order.

        The basis functions are
        :math:`e^{i(\mathbf k+\mathbf Q)\cdot\mathbf r}`. A stored Fourier
        coefficient ``B_g`` therefore enters block ``(Qi, Qj)`` when
        ``Qi - Qj = g``; the reverse block receives ``B_g^\dagger``.

        Returns:
            Dense Hamiltonian in flattened ``Q -> internal`` order.

        Raises:
            ValueError: If momentum, Q vectors, or matching tolerance are invalid.
        """
        momentum = np.asarray(momentum)
        q_vectors = np.asarray(q_vectors)
        if momentum.shape != (2,):
            raise ValueError("momentum must have shape (2,).")
        if q_vectors.ndim != 2 or q_vectors.shape[1:] != (2,):
            raise ValueError("q_vectors must have shape (nq, 2).")
        if matching_tolerance <= 0:
            raise ValueError("matching_tolerance must be positive.")

        nq = len(q_vectors)
        n_internal = self.internal_dim
        result = np.zeros(
            (nq, n_internal, nq, n_internal),
            dtype=np.result_type(
                self.momentum_matrices,
                self.fourier_matrices,
                np.complex64,
            ),
        )
        diagonal = self.momentum_hamiltonian(momentum + q_vectors)
        indices = np.arange(nq)
        result[indices, :, indices, :] = diagonal

        differences = q_vectors[:, None, :] - q_vectors[None, :, :]
        for vector, matrix in zip(
            self.fourier_vectors, self.fourier_matrices, strict=True
        ):
            forward = (
                np.linalg.norm(differences - vector, axis=-1) <= matching_tolerance
            )
            backward = (
                np.linalg.norm(differences + vector, axis=-1) <= matching_tolerance
            )
            result += forward[:, None, :, None] * matrix[None, :, None, :]
            result += backward[:, None, :, None] * matrix.conj().T[None, :, None, :]
        return result.reshape(nq * n_internal, nq * n_internal)

    def save(self, path: str | Path | UPath) -> None:
        """Save the model in the canonical matrix-continuum NPZ format."""
        metadata = {
            "format": "jaqmc.moire.matrix_continuum",
            "version": 1,
            "name": self.name,
            "basis_labels": list(self.basis_labels),
            "energy_unit": self.energy_unit,
            "length_unit": self.length_unit,
            "plane_wave_convention": "exp(i*(k+Q).r)",
        }
        with UPath(path).open("wb") as output:
            np.savez(
                output,
                metadata=np.asarray(json.dumps(metadata)),
                momentum_powers=self.momentum_powers,
                momentum_matrices=self.momentum_matrices,
                fourier_vectors=self.fourier_vectors,
                fourier_matrices=self.fourier_matrices,
            )

    @classmethod
    def load(cls, path: str | Path | UPath) -> "MatrixContinuumModel":
        """Load a canonical matrix-continuum model archive.

        Returns:
            Validated model loaded from ``path``.

        Raises:
            ValueError: If the archive is incomplete or uses an unsupported format.
        """
        with (
            UPath(path).open("rb") as model_file,
            np.load(model_file, allow_pickle=False) as data,
        ):
            try:
                metadata: dict[str, Any] = json.loads(str(data["metadata"].item()))
                if metadata.get("format") != "jaqmc.moire.matrix_continuum":
                    raise ValueError
                if metadata.get("version") != 1:
                    raise ValueError
                if metadata.get("plane_wave_convention") != "exp(i*(k+Q).r)":
                    raise ValueError
                return cls(
                    momentum_powers=data["momentum_powers"],
                    momentum_matrices=data["momentum_matrices"],
                    fourier_vectors=data["fourier_vectors"],
                    fourier_matrices=data["fourier_matrices"],
                    basis_labels=tuple(metadata["basis_labels"]),
                    name=metadata["name"],
                    energy_unit=metadata["energy_unit"],
                    length_unit=metadata["length_unit"],
                )
            except (
                KeyError,
                TypeError,
                ValueError,
                AttributeError,
                IndexError,
            ) as exc:
                raise ValueError(
                    f"Invalid matrix-continuum model file: {path}"
                ) from exc
