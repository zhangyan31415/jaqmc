# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Static structure-factor estimators for periodic systems."""

from collections.abc import Mapping
from dataclasses import field
from typing import Any

from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator, complex_mean_reduce
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.supercell import get_reciprocal_vectors
from jaqmc.utils.wiring import runtime_dep


@configurable_dataclass
class PeriodicStructureFactor(PerWalkerEstimator):
    r"""Charge-density modes and static structure factors in a periodic cell.

    For reciprocal vector :math:`\mathbf q`, this estimator computes

    .. math::

        \rho(\mathbf q) = \sum_i e^{i\mathbf q\cdot\mathbf r_i},\qquad
        S(\mathbf q) = \frac{|\rho(\mathbf q)|^2}{N}.

    ``reciprocal_indices`` specifies :math:`\mathbf q` as integer combinations
    of the simulation-cell reciprocal basis. The zero vector is rejected
    because its structure factor is the trivial constant :math:`N`.

    Args:
        reciprocal_indices: Integer reciprocal coordinates with shape
            ``(n_q, ndim)``.
        lattice: Simulation-cell lattice with row-vector convention.
        data_field: Coordinate field in the sampled data.
        structure_factor_key: Output key for :math:`S(\mathbf q)`.
        density_mode_key: Output key for :math:`\rho(\mathbf q)/N`.
    """

    reciprocal_indices: list[list[int]] = field(default_factory=list)
    structure_factor_key: str = "observable:structure_factor"
    density_mode_key: str = "observable:density_mode"
    lattice: jnp.ndarray = runtime_dep()
    data_field: str = runtime_dep(default="positions")

    def init(self, data: Data, rngs: PRNGKey) -> None:
        """Validate reciprocal coordinates and construct Cartesian vectors.

        Raises:
            ValueError: If lattice or reciprocal-index shapes are invalid.
        """
        del data, rngs
        lattice = jnp.asarray(self.lattice)
        indices = jnp.asarray(self.reciprocal_indices)
        if lattice.ndim != 2 or lattice.shape[0] != lattice.shape[1]:
            raise ValueError("lattice must be a square rank-2 array.")
        if (
            indices.ndim != 2
            or indices.shape[1] != lattice.shape[0]
            or not jnp.issubdtype(indices.dtype, jnp.integer)
        ):
            raise ValueError(
                "reciprocal_indices must be an integer array with shape (n_q, ndim)."
            )
        if len(indices) == 0:
            raise ValueError("reciprocal_indices must contain at least one vector.")
        if bool(jnp.any(jnp.all(indices == 0, axis=1))):
            raise ValueError("reciprocal_indices must not contain the zero vector.")
        self._q_vectors = indices @ get_reciprocal_vectors(lattice)
        return None

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        """Evaluate density modes and structure factors for one walker.

        Returns:
            Density modes and static structure factors at all configured vectors.

        Raises:
            ValueError: If sampled coordinates have incompatible shape.
        """
        del params, prev_walker_stats, rngs
        positions = data[self.data_field]
        if positions.ndim != 2 or positions.shape[1] != self._q_vectors.shape[1]:
            raise ValueError(
                f"{self.data_field} must have shape (n_electron, "
                f"{self._q_vectors.shape[1]})."
            )
        density = jnp.sum(jnp.exp(1j * (positions @ self._q_vectors.T)), axis=0)
        nelec = positions.shape[0]
        return {
            self.structure_factor_key: jnp.abs(density) ** 2 / nelec,
            self.density_mode_key: density / nelec,
        }, state

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        """Return structure statistics and separate complex-mode variances."""
        return complex_mean_reduce(walker_stats)
