# Copyright (c) 2026 JaQMC contributors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in process-local discrete layer basis for the unchanged Moire network.

Call install_representation BEFORE constructing a workflow. Parameters retain
all original shapes. Float-valued spin_coords contain exact labels 0 or 1;
physical-spin sectors remain nspins and are never changed here.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaqmc.app.moire.hamiltonian import _moire_valley_momenta, _physical_spin_signs
from jaqmc.app.moire.wavefunction import MoireWavefunction
from jaqmc.estimator.base import PerWalkerEstimator
from jaqmc.estimator.kinetic import EuclideanKinetic
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep

from .folx_compat import install as _install_sparse_compat
from .mote2 import AbsorbedValleyCorrection
from .mote2 import _install_dense as _dense_install

_ORIGINAL_COMPONENTS = MoireWavefunction.get_layer_components


def validate_labels(labels):
    """Validate a fresh or restored state on the host before tracing.

    Raises:
        ValueError: If labels contain nonfinite values or values other than 0/1.
    """
    labels = np.asarray(labels)
    if not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError(
            "Discrete layer spin_coords must contain only exact 0/1 labels."
        )


def _guard(wavefunction):
    if (
        wavefunction.nlayers != 2
        or wavefunction.ndets != 1
        or wavefunction.phase_mode != "singlephase"
    ):
        raise ValueError("Discrete backend requires singlephase bilayer ndets=1.")


def _discrete_components(self, data):
    _guard(self)
    phi = jnp.stack(self._get_singlephase_layer_orbitals(data), axis=0)
    chi = jax.nn.one_hot(data.spin_coords, 2, dtype=phi.dtype)
    phi0 = jnp.einsum("il,ldij->dij", chi, phi)
    return phi, chi, phi0


def install_representation():
    """Patch the Flax module class before any native closures are constructed."""
    MoireWavefunction.get_layer_components = _discrete_components


@configurable_dataclass
class DiscreteCovariantKinetic(PerWalkerEstimator):
    f_log_psi: Any = runtime_dep()
    primitive_lattice: Any = runtime_dep()
    nspins: tuple[int, int] = runtime_dep()
    prefactor: float = 1.0
    mode: str = "forward_laplacian"
    sparse: bool = True

    def evaluate_single_walker(self, params, data, prev_walker_stats, state, rngs):
        anchor = data.positions
        kb, kt = _moire_valley_momenta(self.primitive_lattice)
        # Labels stay fixed during position differentiation. Unlike the
        # continuous representation there is no auxiliary-angle displacement.
        k_layers = jnp.stack((kb, kt), axis=0)
        k_selected = jax.nn.one_hot(data.spin_coords, 2, dtype=anchor.dtype) @ k_layers
        eta = _physical_spin_signs(self.nspins, anchor.dtype)

        def dressed_logpsi(network_params, inner_data):
            delta = inner_data.positions - anchor
            phase = -1j * jnp.sum(eta[:, None] * k_selected * delta)
            return self.f_log_psi(network_params, inner_data) + phase

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


def install_backend(stage, wavefunction, chunk=512, sparse=True):
    """Install the discrete covariant estimator with the native state layout.

    Returns:
        The modified stage.

    Raises:
        RuntimeError: If the discrete representation was not selected first.
        ValueError: If the wavefunction or chunk is outside the verified scope.
    """
    _guard(wavefunction)
    if MoireWavefunction.get_layer_components is not _discrete_components:
        raise RuntimeError("Call install_representation before workflow construction.")
    if chunk != 512:
        raise ValueError("Comparison backend requires chunk=512.")
    original = stage.estimators.estimators
    kinetic, soc = original["kinetic"], original["soc"]
    kwargs = dict(
        f_log_psi=wavefunction.logpsi,
        primitive_lattice=soc.primitive_lattice,
        nspins=soc.nspins,
        prefactor=kinetic.prefactor,
        vmap_chunk_size=chunk,
    )
    if "grads" in original:
        _dense_install(stage, wavefunction, chunk)
    else:
        original["soc"] = AbsorbedValleyCorrection()
        original["total"].components = [
            "energy:kinetic_plus_soc",
            "energy:coulomb",
            "energy:moire_potential",
        ]
        for estimator in original.values():
            if hasattr(estimator, "vmap_chunk_size"):
                estimator.vmap_chunk_size = chunk
    if sparse:
        _install_sparse_compat()
    original["kinetic"] = DiscreteCovariantKinetic(**kwargs, sparse=sparse)
    return stage


def restore_continuous_representation():
    """Restore native class dispatch before constructing a continuous workflow.

    Existing traced workflows are unaffected; use one representation per process.
    """
    MoireWavefunction.get_layer_components = _ORIGINAL_COMPONENTS
