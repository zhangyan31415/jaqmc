# Copyright (c) 2026 JaQMC contributors
# SPDX-License-Identifier: Apache-2.0

"""Exact bilayer kinetic + valley shift via one covariant Laplacian.

For chi_l(s)=exp(+/- i s), shifting s by -eta*(K_b-K_t).dx/2 and
adding phase -eta*(K_b+K_t).dx/2 implements exp(-i eta K_l.dx).
The ordinary Laplacian of this locally dressed wavefunction at dx=0 is
sum_i (grad_i-i eta_i K_layer)^2 Psi: native kinetic plus native SOC.
The anchor is an outer closure constant, NOT a differentiated coordinate.
This preserves the original wavefunction, Hamiltonian and sampled positions.
"""

from typing import Any

import jax.numpy as jnp

from jaqmc.app.moire.hamiltonian import _moire_valley_momenta, _physical_spin_signs
from jaqmc.estimator.base import PerWalkerEstimator
from jaqmc.estimator.kinetic import EuclideanKinetic
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep


@configurable_dataclass
class CovariantKinetic(PerWalkerEstimator):
    f_log_psi: Any = runtime_dep()
    primitive_lattice: Any = runtime_dep()
    nspins: tuple[int, int] = runtime_dep()
    prefactor: float = 1.0
    mode: str = "forward_laplacian"
    sparse: bool = False

    def evaluate_single_walker(self, params, data, prev_walker_stats, state, rngs):
        anchor = data.positions
        k_bottom, k_top = _moire_valley_momenta(self.primitive_lattice)
        k_mean = (k_bottom + k_top) / 2
        k_difference = (k_bottom - k_top) / 2
        eta = _physical_spin_signs(self.nspins, anchor.dtype)

        def dressed_logpsi(network_params, inner_data):
            displacement = inner_data.positions - anchor
            shifted_angles = inner_data.spin_coords - eta * (
                displacement @ k_difference
            )
            phase = -1j * jnp.sum(eta * (displacement @ k_mean))
            return (
                self.f_log_psi(
                    network_params, inner_data.merge({"spin_coords": shifted_angles})
                )
                + phase
            )

        estimator = EuclideanKinetic(
            f_log_psi=dressed_logpsi,
            data_field="positions",
            mode=self.mode,
            sparse=self.sparse,
            prefactor=self.prefactor,
        )
        answer, _ = estimator.evaluate_single_walker(
            params, data, prev_walker_stats, state, rngs
        )
        return {"energy:kinetic_plus_soc": answer["energy:kinetic"]}, state
