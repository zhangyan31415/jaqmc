# Copyright (c) 2026 JaQMC contributors
# SPDX-License-Identifier: Apache-2.0

"""Optional exact bilayer adapters for integer-filling MoTe2 calculations.

Native workflows remain the default. Select the representation before workflow
construction, install this backend after it, then initialize binary labels and
install the binary sampler before burn-in when using the discrete variant.
The representation switch patches a Flax class in this process; do not mix
already-traced continuous and discrete workflows in the same process.
"""

import jax.numpy as jnp
from kfac_jax import utils as kfac_utils

from jaqmc.estimator.base import PerWalkerEstimator

from .covariant import CovariantKinetic
from .sampling import initialize_labels as initialize_labels
from .sampling import install as install_sampler
from .shared import WeightedLossAndGrad

__all__ = [
    "initialize_labels",
    "install_backend",
    "install_sampler",
    "prepare_representation",
]


class AbsorbedValleyCorrection(PerWalkerEstimator):
    """Retain the native SOC state key; its energy is inside covariant kinetic."""

    def evaluate_single_walker(self, params, data, prev_walker_stats, state, rngs):
        del params, data, prev_walker_stats, rngs
        return {
            "valley_correction_absorbed": jnp.asarray(0.0, dtype=jnp.float32)
        }, state


def _install_dense(stage, wavefunction, chunk=512):
    if wavefunction.nlayers != 2:
        raise ValueError(
            "The covariant angle-shift backend requires exactly two layers."
        )
    kfac_utils.set_use_cholesky_inversion(True)
    estimators = stage.estimators.estimators
    original_keys = set(estimators)
    kinetic, soc = estimators["kinetic"], estimators["soc"]
    estimators["kinetic"] = CovariantKinetic(
        f_log_psi=wavefunction.logpsi,
        primitive_lattice=soc.primitive_lattice,
        nspins=soc.nspins,
        prefactor=kinetic.prefactor,
        vmap_chunk_size=chunk,
    )
    estimators["soc"] = AbsorbedValleyCorrection()
    estimators["total"].components = [
        "energy:kinetic_plus_soc",
        "energy:coulomb",
        "energy:moire_potential",
    ]
    if "grads" in estimators:
        grads = estimators["grads"]
        estimators["grads"] = WeightedLossAndGrad(
            f_log_psi=grads.f_log_psi,
            loss_key=grads.loss_key,
            clip_method=grads.clip_method,
            clip_scale=grads.clip_scale,
        )
    for name, estimator in estimators.items():
        if name != "grads" and hasattr(estimator, "vmap_chunk_size"):
            estimator.vmap_chunk_size = chunk
    if set(estimators) != original_keys:
        raise AssertionError(
            "The optimized backend changed native estimator state keys."
        )
    return stage


def prepare_representation(representation="continuous"):
    """Select process-local class dispatch BEFORE constructing native closures.

    Explicit ``continuous`` restores the native method. It does not convert
    existing binary labels to angles or invalidate compiled functions.

    Raises:
        ValueError: If the requested representation is unknown.
    """
    from .discrete import install_representation, restore_continuous_representation

    if representation == "continuous":
        restore_continuous_representation()
    elif representation == "discrete":
        install_representation()
    else:
        raise ValueError("representation must be 'continuous' or 'discrete'")


def install_backend(
    stage, wavefunction, chunk=512, representation="continuous", sparse=True
):
    """Install merged kinetic/SOC, weighted gradients and Cholesky KFAC.

    Verified sparse/discrete scope is a bilayer singlephase single determinant,
    chunk512. The continuous dense variant allows other positive chunk sizes.
    These restrictions fail explicitly rather than silently changing algorithms.
    Native checkpoint and optimizer state structures are retained.

    Returns:
        The modified native stage.

    Raises:
        ValueError: If the representation, model or chunk is unsupported.
        RuntimeError: If representation preparation was omitted.
    """
    if representation not in ("continuous", "discrete"):
        raise ValueError("representation must be 'continuous' or 'discrete'")
    if chunk < 1:
        raise ValueError("chunk must be positive")
    if wavefunction.nlayers != 2:
        raise ValueError("The optimized MoTe2 backend requires exactly two layers.")
    if sparse or representation == "discrete":
        if wavefunction.ndets != 1 or wavefunction.phase_mode != "singlephase":
            raise ValueError(
                "Verified sparse/discrete backend requires singlephase bilayer ndets=1."
            )
        if chunk != 512:
            raise ValueError("Verified sparse/discrete backend requires chunk=512.")
    from jaqmc.app.moire.wavefunction import MoireWavefunction

    from .discrete import _discrete_components

    if representation == "discrete":
        from .discrete import install_backend as install_discrete

        return install_discrete(stage, wavefunction, chunk=chunk, sparse=sparse)
    if MoireWavefunction.get_layer_components is _discrete_components:
        raise RuntimeError(
            "Call prepare_representation('continuous') before workflow construction."
        )
    _install_dense(stage, wavefunction, chunk)
    if sparse:
        from .folx_compat import install

        install()
        stage.estimators.estimators["kinetic"].sparse = True
    return stage
