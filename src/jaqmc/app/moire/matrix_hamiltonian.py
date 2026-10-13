# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Estimators for material-independent matrix continuum Hamiltonians."""

from collections.abc import Callable, Mapping
from dataclasses import field
from typing import Any

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.estimator.base import PerWalkerEstimator
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep

from .data import MoireData
from .hamiltonian import (
    _build_exchanged_orbitals,
    compute_operator_derivative_ratio,
    compute_operator_ratio,
)

type InternalComponentsFn = Callable[
    [Params, MoireData], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
]


@configurable_dataclass
class MatrixContinuumObservable(PerWalkerEstimator):
    r"""Local estimators for constant one-body internal-space operators.

    For each configured matrix :math:`A_\alpha`, this estimator evaluates

    .. math::

        \frac{1}{N}\sum_i
        \frac{A_\alpha^{(i)}\Psi}{\Psi}.

    All operators share one internal-component wavefunction evaluation. This is
    useful for layer, orbital, spin, valley, or other finite-basis
    polarizations. Complex matrices are configured as separate real and
    imaginary arrays so they remain portable in YAML.

    Args:
        operator_matrices_real: Real parts with shape
            ``(n_operators, n_internal, n_internal)``.
        operator_matrices_imag: Optional imaginary parts with the same shape.
        output_keys: Statistic key for each operator.
        normalize_by_electron: Divide each extensive one-body sum by the
            electron count.
        f_internal_components: Function returning ``(component_orbitals,
            amplitudes, identity_orbitals)``.
    """

    operator_matrices_real: list = field(default_factory=list)
    operator_matrices_imag: list = field(default_factory=list)
    output_keys: list[str] = field(default_factory=list)
    normalize_by_electron: bool = True
    f_internal_components: InternalComponentsFn = runtime_dep()

    def init(self, data: MoireData, rngs: PRNGKey) -> None:
        """Validate and assemble the configured Hermitian operators.

        Raises:
            ValueError: If operator arrays or output keys are invalid.
        """
        del data, rngs
        real = jnp.asarray(self.operator_matrices_real)
        imag = (
            jnp.asarray(self.operator_matrices_imag)
            if len(self.operator_matrices_imag)
            else jnp.zeros_like(real)
        )
        if real.ndim != 3 or real.shape[1] != real.shape[2]:
            raise ValueError(
                "operator_matrices_real must have shape "
                "(n_operators, n_internal, n_internal)."
            )
        if imag.shape != real.shape:
            raise ValueError(
                "operator_matrices_imag must have the same shape as "
                "operator_matrices_real."
            )
        if len(self.output_keys) != real.shape[0]:
            raise ValueError(
                "output_keys must contain one key per internal operator matrix."
            )
        if len(set(self.output_keys)) != len(self.output_keys):
            raise ValueError("output_keys must be unique.")
        matrices = real + 1j * imag
        if not bool(jnp.all(jnp.isfinite(matrices))):
            raise ValueError("Internal observable matrices must be finite.")
        if not bool(jnp.allclose(matrices, matrices.conj().swapaxes(-1, -2))):
            raise ValueError("Internal observable matrices must be Hermitian.")
        self._operator_matrices = matrices
        return None

    def evaluate_single_walker(
        self,
        params: Params,
        data: MoireData,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        """Evaluate all configured internal one-body observables.

        Returns:
            Mapping from configured output keys to local estimator values.

        Raises:
            ValueError: If wavefunction and operator internal dimensions differ.
        """
        del prev_walker_stats, rngs
        component_orbitals, amplitudes, identity_orbitals = self.f_internal_components(
            params, data
        )
        n_internal = self._operator_matrices.shape[1]
        if component_orbitals.shape[0] != n_internal:
            raise ValueError(
                "Wavefunction component count does not match the internal "
                f"observable matrices: {component_orbitals.shape[0]} != {n_internal}."
            )
        if amplitudes.shape[-1] != n_internal:
            raise ValueError(
                "Internal-coordinate amplitude count does not match the internal "
                f"observable matrices: {amplitudes.shape[-1]} != {n_internal}."
            )

        operator_orbitals = jnp.einsum(
            "ia,oab,bdij->odij",
            amplitudes,
            self._operator_matrices,
            component_orbitals,
        )
        values = jnp.sum(
            compute_operator_ratio(identity_orbitals, operator_orbitals), axis=-1
        )
        if self.normalize_by_electron:
            values = values / data.positions.shape[0]
        return {
            output_key: values[index]
            for index, output_key in enumerate(self.output_keys)
        }, state


def _scaled_operator_numerator_fn(
    params: Params,
    data: MoireData,
    operator_orbitals_fn: Callable[
        [Params, MoireData], tuple[jnp.ndarray, jnp.ndarray]
    ],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build numerically scaled row-replacement numerators.

    The returned function has shape ``(noperator, nelec)`` and is normalized by
    the untransformed wavefunction at the evaluation point. The denominator is
    held constant under spatial differentiation, so its derivatives are exactly
    those of the operator numerator divided by ``psi``.

    Returns:
        Function mapping flattened positions to scaled operator numerators.
    """
    nelec, ndim = data.positions.shape
    base_orbitals, _ = operator_orbitals_fn(params, data)
    base_sign, base_logabs = jnp.linalg.slogdet(base_orbitals)
    log_scale = jax.lax.stop_gradient(jnp.max(base_logabs))
    denominator = jax.lax.stop_gradient(
        jnp.sum(base_sign * jnp.exp(base_logabs - log_scale))
    )
    eye = jnp.eye(nelec)
    ones = jnp.ones((nelec, nelec))
    replace_row = jax.vmap(jnp.outer)(eye, ones)
    exchange_masks = (replace_row, ones - replace_row)

    def numerator_ratio(spatial_flat: jnp.ndarray) -> jnp.ndarray:
        positions = spatial_flat.reshape(nelec, ndim)
        base, transformed = operator_orbitals_fn(
            params, data.merge({"positions": positions})
        )
        exchanged = _build_exchanged_orbitals(
            base_orbitals=base,
            operator_orbitals=transformed,
            exchange_masks=exchange_masks,
        )
        sign, logabs = jnp.linalg.slogdet(jnp.moveaxis(exchanged, -1, -3))
        numerator = jnp.sum(sign * jnp.exp(logabs - log_scale), axis=1)
        return numerator / denominator

    return numerator_ratio


def _operator_second_derivative_ratio(
    params: Params,
    data: MoireData,
    *,
    operator_orbitals_fn: Callable[
        [Params, MoireData], tuple[jnp.ndarray, jnp.ndarray]
    ],
    axes: tuple[int, int],
) -> jnp.ndarray:
    """Compute one same-electron Hessian component of ``O_i psi / psi``.

    Returns:
        Ratios shaped ``(noperator, nelec)``.

    Raises:
        ValueError: If positions are not two-dimensional.
    """
    nelec, ndim = data.positions.shape
    if ndim != 2:
        raise ValueError("Matrix continuum differential terms require 2D positions.")
    numerator_ratio = _scaled_operator_numerator_fn(params, data, operator_orbitals_fn)
    positions_flat = data.positions.reshape(-1)
    basis = parallel_jax.pvary(jnp.eye(positions_flat.size))

    def body(result: jnp.ndarray, electron_idx: jnp.ndarray):
        first_direction = basis[electron_idx * ndim + axes[0]]
        second_direction = basis[electron_idx * ndim + axes[1]]

        def first_directional(spatial_flat: jnp.ndarray) -> jnp.ndarray:
            return jax.jvp(
                numerator_ratio,
                (spatial_flat,),
                (first_direction,),
            )[1]

        second = jax.jvp(
            first_directional,
            (positions_flat,),
            (second_direction,),
        )[1]
        return result.at[:, electron_idx].set(second[:, electron_idx]), None

    initial = numerator_ratio(positions_flat)
    result, _ = jax.lax.scan(
        body,
        jnp.zeros_like(initial),
        jnp.arange(nelec, dtype=jnp.int32),
    )
    return result


@configurable_dataclass
class MatrixContinuumPotential(PerWalkerEstimator):
    r"""Local matrix potential for a generic continuum model.

    The internal-space potential is

    .. math::

        V(\mathbf r) = A_0 + \sum_g
        \left(B_g e^{i\mathbf g\cdot\mathbf r}
        + B_g^\dagger e^{-i\mathbf g\cdot\mathbf r}\right).

    ``f_internal_components`` returns component orbitals, continuous internal
    coordinate amplitudes, and the corresponding identity orbitals. Components
    may represent any flattened combination of layer, orbital, spin, valley, or
    other finite internal degrees of freedom.

    Args:
        f_internal_components: Function returning ``(component_orbitals,
            amplitudes, identity_orbitals)``.
        constant_matrix: Hermitian zero-momentum matrix ``A0``.
        fourier_vectors: One vector from each nonzero ``+/-g`` pair.
        fourier_matrices: One-sided Fourier coefficients ``Bg``.
        output_key: Output statistics key.
    """

    f_internal_components: Callable[
        [Params, MoireData], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ] = runtime_dep()
    constant_matrix: jnp.ndarray = runtime_dep()
    fourier_vectors: jnp.ndarray = runtime_dep()
    fourier_matrices: jnp.ndarray = runtime_dep()
    output_key: str = "energy:matrix_potential"

    def init(self, data: MoireData, rngs: PRNGKey) -> None:
        """Validate model-array shapes before estimator compilation.

        Returns:
            No estimator state.

        Raises:
            ValueError: If the matrix-continuum arrays have incompatible shapes.
        """
        del data, rngs
        self.constant_matrix = jnp.asarray(self.constant_matrix)
        self.fourier_vectors = jnp.asarray(self.fourier_vectors)
        self.fourier_matrices = jnp.asarray(self.fourier_matrices)
        if (
            self.constant_matrix.ndim != 2
            or self.constant_matrix.shape[0] != self.constant_matrix.shape[1]
        ):
            raise ValueError("constant_matrix must be a square rank-2 array.")
        n_internal = self.constant_matrix.shape[0]
        if self.fourier_vectors.ndim != 2 or self.fourier_vectors.shape[1:] != (2,):
            raise ValueError("fourier_vectors must have shape (n_terms, 2).")
        if self.fourier_matrices.shape != (
            len(self.fourier_vectors),
            n_internal,
            n_internal,
        ):
            raise ValueError(
                "fourier_matrices must have shape (n_terms, n_internal, n_internal)."
            )
        return None

    def evaluate_single_walker(
        self,
        params: Params,
        data: MoireData,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        """Evaluate the local matrix-potential energy for one walker.

        Returns:
            Potential-energy statistic and unchanged estimator state.

        Raises:
            ValueError: If wavefunction and Hamiltonian internal dimensions differ.
        """
        del prev_walker_stats, rngs
        component_orbitals, amplitudes, identity_orbitals = self.f_internal_components(
            params, data
        )
        n_internal = self.constant_matrix.shape[0]
        if component_orbitals.shape[0] != n_internal:
            raise ValueError(
                "Wavefunction component count does not match the continuum model: "
                f"{component_orbitals.shape[0]} != {n_internal}."
            )
        if amplitudes.shape[-1] != n_internal:
            raise ValueError(
                "Internal-coordinate amplitude count does not match the continuum "
                f"model: {amplitudes.shape[-1]} != {n_internal}."
            )

        phase = jnp.exp(1j * (data.positions @ self.fourier_vectors.T))
        forward = jnp.einsum("ig,gab->iab", phase, self.fourier_matrices)
        potential = (
            self.constant_matrix[None, ...]
            + forward
            + jnp.swapaxes(forward.conj(), -1, -2)
        )
        operator_orbitals = jnp.einsum(
            "ia,iab,bdij->dij",
            amplitudes,
            potential,
            component_orbitals,
        )
        energy = jnp.sum(compute_operator_ratio(identity_orbitals, operator_orbitals))
        return {self.output_key: energy}, state


@configurable_dataclass
class MatrixContinuumDifferential(PerWalkerEstimator):
    r"""Local estimator for matrix-valued momentum-polynomial terms.

    Each term applies an internal matrix and then the Cartesian momentum
    monomial

    .. math::

        p_x^{n_x}p_y^{n_y} A,\qquad p_\alpha=-i\partial_\alpha.

    The initial implementation supports total degree one or two. Same-electron
    Hessian components are evaluated with directional JVPs, so the estimator
    does not materialize the full ``(2 * nelec, 2 * nelec)`` Hessian.

    Args:
        f_internal_components: Function returning ``(component_orbitals,
            amplitudes, identity_orbitals)``.
        momentum_powers: Nonconstant powers ``(nx, ny)``.
        momentum_matrices: Internal coefficient matrix for every power.
        output_key: Output statistics key.
    """

    f_internal_components: InternalComponentsFn = runtime_dep()
    momentum_powers: jnp.ndarray = runtime_dep()
    momentum_matrices: jnp.ndarray = runtime_dep()
    output_key: str = "energy:matrix_differential"

    def init(self, data: MoireData, rngs: PRNGKey) -> None:
        """Validate differential-term shapes and supported derivative orders.

        Raises:
            ValueError: If term arrays are malformed or contain unsupported powers.
        """
        del data, rngs
        self.momentum_powers = jnp.asarray(self.momentum_powers)
        self.momentum_matrices = jnp.asarray(self.momentum_matrices)
        if self.momentum_powers.ndim != 2 or self.momentum_powers.shape[1:] != (2,):
            raise ValueError("momentum_powers must have shape (n_terms, 2).")
        if not jnp.issubdtype(self.momentum_powers.dtype, jnp.integer):
            raise ValueError("momentum_powers must contain integers.")
        powers = tuple(
            tuple(int(component) for component in power)
            for power in self.momentum_powers.tolist()
        )
        if any(min(power) < 0 or sum(power) not in (1, 2) for power in powers):
            raise ValueError(
                "MatrixContinuumDifferential supports non-negative powers "
                "with total degree one or two."
            )
        if (
            self.momentum_matrices.ndim != 3
            or self.momentum_matrices.shape[0] != len(powers)
            or self.momentum_matrices.shape[1] != self.momentum_matrices.shape[2]
        ):
            raise ValueError(
                "momentum_matrices must have shape (n_terms, n_internal, n_internal)."
            )
        self._powers = powers
        self._has_first_order = any(sum(power) == 1 for power in powers)
        return None

    def evaluate_single_walker(
        self,
        params: Params,
        data: MoireData,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        """Evaluate all configured momentum-polynomial terms.

        Returns:
            Differential-energy statistic and unchanged estimator state.

        Raises:
            ValueError: If wavefunction and Hamiltonian internal dimensions differ.
        """
        del prev_walker_stats, rngs
        component_orbitals, amplitudes, _ = self.f_internal_components(params, data)
        n_internal = self.momentum_matrices.shape[1]
        if component_orbitals.shape[0] != n_internal:
            raise ValueError(
                "Wavefunction component count does not match the continuum model: "
                f"{component_orbitals.shape[0]} != {n_internal}."
            )
        if amplitudes.shape[-1] != n_internal:
            raise ValueError(
                "Internal-coordinate amplitude count does not match the continuum "
                f"model: {amplitudes.shape[-1]} != {n_internal}."
            )

        def operator_orbitals_fn(
            network_params: Params, walker_data: MoireData
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            orbitals, internal_amplitudes, identity = self.f_internal_components(
                network_params, walker_data
            )
            transformed = jnp.einsum(
                "ia,tab,bdij->tdij",
                internal_amplitudes,
                self.momentum_matrices,
                orbitals,
            )
            return identity, transformed

        first_derivatives = (
            compute_operator_derivative_ratio(
                params,
                data,
                operator_orbitals_fn=operator_orbitals_fn,
            )
            if self._has_first_order
            else None
        )
        second_derivatives: dict[tuple[int, int], jnp.ndarray] = {}
        energy = jnp.asarray(0.0, dtype=component_orbitals.dtype)
        for term_idx, power in enumerate(self._powers):
            degree = sum(power)
            if degree == 1:
                assert first_derivatives is not None
                axis = 0 if power == (1, 0) else 1
                ratio = first_derivatives[term_idx, :, axis]
            else:
                axes = (
                    (0, 0) if power == (2, 0) else (1, 1) if power == (0, 2) else (0, 1)
                )
                if axes not in second_derivatives:
                    second_derivatives[axes] = _operator_second_derivative_ratio(
                        params,
                        data,
                        operator_orbitals_fn=operator_orbitals_fn,
                        axes=axes,
                    )
                ratio = second_derivatives[axes][term_idx]
            energy = energy + (-1j) ** degree * jnp.sum(ratio)
        return {self.output_key: energy}, state
