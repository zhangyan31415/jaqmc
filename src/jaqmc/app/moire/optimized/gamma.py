# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Gamma matrix Hamiltonian in the native VMC estimator pipeline.

Each estimator owns its derivative method. Installing another estimator never
replaces a module-global energy function. ``baseline`` retains the sequential
three-direction scan; ``jvp``/``lap``/``sparse-lap`` group directional jets.
"""

import math
from typing import Any

import jax.numpy as jnp
from kfac_jax import utils as kfac_utils

from jaqmc.estimator import EstimatorPipeline
from jaqmc.estimator.base import PerWalkerEstimator
from jaqmc.estimator.total_energy import TotalEnergy
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep

from .gamma_directions import kinetic_grouped
from .gamma_operator import kinetic_analytic
from .shared import CenteredLossAndGrad, WeightedLossAndGrad


@configurable_dataclass
class GammaMatrixEnergy(PerWalkerEstimator):
    wavefunction: Any = runtime_dep()
    model: Any = runtime_dep()
    length_A: float = runtime_dep()
    spin_signs: Any = runtime_dep()
    derivative_method: str = "jvp"
    directions: int = 6

    def evaluate_single_walker(self, params, data, prev_walker_stats, state, rngs):
        args = (
            self.wavefunction,
            params,
            data.positions,
            data.spin_coords,
            self.spin_signs,
            self.model,
            self.length_A,
        )
        if self.derivative_method == "baseline":
            value = kinetic_analytic(*args, include_potential=True)
        else:
            value = kinetic_grouped(
                *args,
                include_potential=True,
                method=self.derivative_method,
                directions=self.directions,
            )
        return {"energy:gamma": value}, state


def install_backend(
    stage,
    wavefunction,
    model,
    length_A,
    chunk=512,
    optimized=True,
    derivative_method="jvp",
    directions=6,
):
    """Install the full four-component Gamma model, preserving Coulomb/Ewald.

    ``optimized`` selects weighted gradients and Cholesky KFAC independently
    of the derivative method. All methods retain the complete a1/a2 kinetic
    operator, mixed spatial derivatives, and cross-particle backflow.

    Returns:
        The supplied stage with its Gamma estimator pipeline installed.

    Raises:
        ValueError: If the wavefunction or backend configuration is invalid.
    """
    if wavefunction.nlayers != 4:
        raise ValueError(
            "Gamma layer x orbital basis requires four internal components."
        )
    if derivative_method not in ("baseline", "jvp", "lap", "sparse-lap"):
        raise ValueError(f"Unknown Gamma derivative method: {derivative_method}")
    if (
        not isinstance(directions, int)
        or isinstance(directions, bool)
        or directions < 1
    ):
        raise ValueError("directions must be a positive integer.")
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
        raise ValueError("chunk must be a positive integer.")
    if not math.isfinite(length_A) or length_A <= 0:
        raise ValueError("length_A must be finite and positive.")
    kfac_utils.set_use_cholesky_inversion(optimized)
    native = stage.estimators.estimators
    grad, coulomb = native["grads"], native["coulomb"]
    coulomb.vmap_chunk_size = chunk
    cls = WeightedLossAndGrad if optimized else CenteredLossAndGrad
    signs = jnp.asarray(
        [1.0] * wavefunction.nspins[0] + [-1.0] * wavefunction.nspins[1]
    )
    stage.estimators = EstimatorPipeline(
        {
            "gamma": GammaMatrixEnergy(
                wavefunction=wavefunction,
                model=model,
                length_A=length_A,
                spin_signs=signs,
                vmap_chunk_size=chunk,
                derivative_method=derivative_method,
                directions=directions,
            ),
            "coulomb": coulomb,
            "total": TotalEnergy(components=["energy:gamma", "energy:coulomb"]),
            "grads": cls(
                f_log_psi=grad.f_log_psi,
                clip_method=grad.clip_method,
                clip_scale=grad.clip_scale,
                loss_key=grad.loss_key,
            ),
        }
    )
    return stage
