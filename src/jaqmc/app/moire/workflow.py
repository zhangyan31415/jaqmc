# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Train and evaluation workflows for moire systems."""

from functools import partial
from typing import Any

import jax
from jax import numpy as jnp

from jaqmc.estimator import EstimatorLike
from jaqmc.estimator.kinetic import EuclideanKinetic
from jaqmc.estimator.loss_grad import LossAndGrad
from jaqmc.estimator.total_energy import TotalEnergy
from jaqmc.geometry.pbc import wrap_positions
from jaqmc.optimizer.kfac import KFACOptimizer
from jaqmc.sampler.mcmc import MCMCSampler, SamplingProposal
from jaqmc.utils.config import (
    ConfigManager,
    ConfigManagerLike,
    configurable_dataclass,
)
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction import Wavefunction
from jaqmc.workflow.evaluation import EvaluationWorkflow
from jaqmc.workflow.stage.evaluation import EvaluationWorkStage
from jaqmc.workflow.stage.vmc import VMCWorkStage
from jaqmc.workflow.vmc import VMCWorkflow

from .config import MoireConfig
from .data import data_init
from .hamiltonian import CoulombInteractionEnergy, MoirePotential, MoireSOC
from .matrix_hamiltonian import (
    MatrixContinuumDifferential,
    MatrixContinuumPotential,
)
from .model import MatrixContinuumModel
from .wavefunction import MoireWavefunction


@configurable_dataclass
class MoireJointMCMCSampler(MCMCSampler):
    """Joint MH sampler for moire positions and spin angles.

    Proposes a spatial move (in dimensionless lattice coordinates) and a
    layer-pseudospin angle move together, so both degrees of freedom are
    updated in a single Metropolis-Hastings step.

    Args:
        steps: Number of Metropolis-Hastings updates per sample draw.
        initial_width: Initial stddev of the spatial Gaussian proposal in
            lattice coordinates.
        adapt_frequency: Number of steps between adaptive proposal-width
            updates.
        spin_mass: Relative stiffness of the spin-angle proposal. The
            spin-angle step width is ``2 * pi * stddev / spin_mass``, where
            ``stddev`` is the adaptively tuned spatial width; larger values
            shrink spin moves relative to spatial moves. Defaults to ``1.0``.
    """

    # Moire-tuned proposal: the generic MCMCSampler defaults (width=0.1,
    # steps=10) mix poorly in moire units, stalling early optimization.
    steps: int = 20
    initial_width: float = 0.02
    adapt_frequency: int = 50
    spin_mass: float = 1.0
    sampling_proposal: SamplingProposal = runtime_dep()

    def configure(self, lattice: jnp.ndarray):
        lattice = jnp.asarray(lattice)
        spin_mass = self.spin_mass

        def proposal(
            rngs: jax.Array, x: dict[str, Any], stddev: float | jnp.ndarray
        ) -> dict[str, Any]:
            positions = x["positions"]
            spin_coords = x["spin_coords"]
            rng_spatial, rng_spin = jax.random.split(rngs)
            spatial_noise_lattice = stddev * jax.random.normal(
                rng_spatial, positions.shape
            )
            positions_new = wrap_positions(
                positions + spatial_noise_lattice @ lattice, lattice
            )
            spin_stddev = 2.0 * jnp.pi * stddev / spin_mass
            spin_new = (
                spin_coords
                + jax.random.normal(rng_spin, spin_coords.shape) * spin_stddev
            ) % (2.0 * jnp.pi)
            return {"positions": positions_new, "spin_coords": spin_new}

        self.sampling_proposal = proposal


def load_continuum_model(
    cfg: ConfigManagerLike,
    system_config: MoireConfig,
) -> MatrixContinuumModel | None:
    """Load and convert the configured high-throughput continuum model.

    Returns:
        Model in meV and dimensionless simulation coordinates, or ``None`` when
        ``continuum_model.path`` is empty.
    """
    model_path = cfg.get("continuum_model.path", "")
    if not model_path:
        return None
    return MatrixContinuumModel.load(model_path).to_dimensionless(
        system_config.moire_lattice_constant_nm
    )


def _configure_system_and_model(
    cfg: ConfigManagerLike,
) -> tuple[MoireConfig, MoireWavefunction, MatrixContinuumModel | None]:
    """Build shared system, wavefunction, and optional continuum model.

    Returns:
        Tuple of system configuration, wavefunction, and optional converted model.

    Raises:
        TypeError: If the wavefunction does not implement MoireWavefunction.
    """
    system_config: MoireConfig = cfg.get_module("system", "jaqmc.app.moire.config")
    continuum_model = load_continuum_model(cfg, system_config)

    wf = cfg.get_module("wf", "jaqmc.app.moire.wavefunction")
    if not isinstance(wf, Wavefunction) or not isinstance(wf, MoireWavefunction):
        raise TypeError(
            f"Wavefunction must implement MoireWavefunction, got {type(wf).__name__}."
        )
    wf.nspins = system_config.electron_spins
    wf.primitive_lattice = jnp.asarray(system_config.lattice_vectors)
    wf.simulation_lattice = jnp.asarray(system_config.supercell_lattice)
    wf.twist = jnp.asarray(system_config.twist)
    if continuum_model is not None:
        wf.nlayers = continuum_model.internal_dim
    wf.configure_phase_inputs(system_config)
    return system_config, wf, continuum_model


def configure_system(
    cfg: ConfigManagerLike,
) -> tuple[MoireConfig, MoireWavefunction]:
    """Build the shared system objects for moire workflows.

    Returns:
        Tuple of ``(system_config, wavefunction)``. When
        ``continuum_model.path`` is configured, the model determines the
        wavefunction's internal component count.
    """
    system_config, wf, _ = _configure_system_and_model(cfg)
    return system_config, wf


def make_estimators(
    cfg: ConfigManagerLike,
    wf: MoireWavefunction,
    system_config: MoireConfig,
    *,
    always_enable_energy: bool = False,
    continuum_model: MatrixContinuumModel | None = None,
) -> dict[str, EstimatorLike]:
    estimators: dict[str, EstimatorLike] = {}
    layer_components_fn = wf.layer_components
    moire_lattice = jnp.asarray(system_config.moire_lattice_vectors)

    if always_enable_energy or cfg.get("estimators.enabled.energy", True):
        if continuum_model is not None:
            components = []
            if cfg.get("estimators.enabled.matrix_potential", True):
                estimators["matrix_potential"] = cfg.get(
                    "estimators.energy.matrix_potential",
                    MatrixContinuumPotential(
                        f_internal_components=wf.internal_components,
                        constant_matrix=jnp.asarray(continuum_model.constant_matrix),
                        fourier_vectors=jnp.asarray(continuum_model.fourier_vectors),
                        fourier_matrices=jnp.asarray(continuum_model.fourier_matrices),
                    ),
                )
                components.append("energy:matrix_potential")
            powers, matrices = continuum_model.differential_terms
            if len(powers) and cfg.get("estimators.enabled.matrix_differential", True):
                estimators["matrix_differential"] = cfg.get(
                    "estimators.energy.matrix_differential",
                    MatrixContinuumDifferential(
                        f_internal_components=wf.internal_components,
                        momentum_powers=jnp.asarray(powers),
                        momentum_matrices=jnp.asarray(matrices),
                    ),
                )
                components.append("energy:matrix_differential")
            if cfg.get("estimators.enabled.coulomb", True):
                estimators["coulomb"] = CoulombInteractionEnergy(
                    supercell_lattice=jnp.asarray(system_config.supercell_lattice),
                    coulomb_prefactor_mev=system_config.coulomb_prefactor_mev,
                )
                components.append("energy:coulomb")
            estimators["total"] = TotalEnergy(components=components)
            return estimators

        # Shared factor mapping the dimensionless valley-momentum kinetic
        # operator to meV; used by both the kinetic and SOC estimators.
        kinetic_prefactor = (
            system_config.kinetic_prefactor_mev / system_config.effective_mass
        )
        components = []
        if cfg.get("estimators.enabled.kinetic", True):
            estimators["kinetic"] = cfg.get(
                "estimators.energy.kinetic",
                EuclideanKinetic(
                    f_log_psi=wf.logpsi,
                    data_field="positions",
                    prefactor=kinetic_prefactor,
                ),
            )
            components.append("energy:kinetic")
        if cfg.get("estimators.enabled.coulomb", True):
            estimators["coulomb"] = CoulombInteractionEnergy(
                supercell_lattice=jnp.asarray(system_config.supercell_lattice),
                coulomb_prefactor_mev=system_config.coulomb_prefactor_mev,
            )
            components.append("energy:coulomb")
        if cfg.get("estimators.enabled.moire_potential", True):
            estimators["moire_potential"] = MoirePotential(
                f_layer_components=layer_components_fn,
                primitive_lattice=moire_lattice,
                nspins=system_config.electron_spins,
                v1_mev=system_config.v1_mev,
                phi1_rad=system_config.phi1_rad,
                omega1_mev=system_config.omega1_mev,
            )
            components.append("energy:moire_potential")
        if cfg.get("estimators.enabled.soc", True):
            estimators["soc"] = MoireSOC(
                f_layer_components=layer_components_fn,
                primitive_lattice=moire_lattice,
                nspins=system_config.electron_spins,
                prefactor=kinetic_prefactor,
            )
            components.append("energy:soc")
        estimators["total"] = TotalEnergy(components=components)

    return estimators


class MoireTrainWorkflow(VMCWorkflow):
    """VMC training workflow for moire systems."""

    @classmethod
    def default_preset(cls) -> dict[str, Any]:
        console_fields = (
            "pmove:.2f,energy=total_energy_real:.4f,variance=total_energy_real_var:.4f"
        )
        return {
            "train": {
                "run": {"iterations": 200_000},
                # Moire-tuned KFAC hyperparameters. Presets are merged below
                # user YAML and CLI overrides, so every key stays overridable.
                "optim": {
                    "learning_rate": {
                        "module": "jaqmc.optimizer.schedule:Standard",
                        "rate": 0.003,
                        "delay": 10000.0,
                    },
                    "damping": 0.0003,
                },
                "grads": {"clip_scale": 20.0},
                "writers": {"console": {"fields": console_fields}},
            },
        }

    def __init__(self, cfg: ConfigManager) -> None:
        super().__init__(cfg)
        system_config, wf, continuum_model = _configure_system_and_model(cfg)
        self.wf = wf
        self.data_init = partial(data_init, system_config)
        sampler = cfg.get("sampler", MoireJointMCMCSampler)
        sampler.configure(jnp.asarray(system_config.supercell_lattice))
        train = VMCWorkStage.builder(cfg.scoped("train"), wf)
        train.configure_sample_plan(wf.logpsi, {("positions", "spin_coords"): sampler})
        # Default to the shared KFAC optimizer; the moire-tuned learning rate
        # and damping come from default_preset, so users can override any field
        # (or the optimizer itself) via train.optim.*.
        train.configure_optimizer(default=KFACOptimizer, f_log_psi=wf.logpsi)
        train.configure_estimators(
            **make_estimators(
                cfg,
                wf,
                system_config,
                always_enable_energy=True,
                continuum_model=continuum_model,
            )
        )
        train.configure_loss_grads(LossAndGrad, f_log_psi=wf.logpsi)
        self.train_stage = train.build()


class MoireEvalWorkflow(EvaluationWorkflow):
    """Evaluation workflow for moire systems."""

    def __init__(self, cfg: ConfigManager) -> None:
        super().__init__(cfg)
        system_config, wf, continuum_model = _configure_system_and_model(cfg)
        self.wf = wf
        self.data_init = partial(data_init, system_config)
        evaluation = EvaluationWorkStage.builder(cfg, wf, name="evaluation")
        sampler = cfg.get("sampler", MoireJointMCMCSampler)
        sampler.configure(jnp.asarray(system_config.supercell_lattice))
        evaluation.configure_sample_plan(
            wf.logpsi, {("positions", "spin_coords"): sampler}
        )
        evaluation.configure_estimators(
            **make_estimators(
                cfg,
                wf,
                system_config,
                continuum_model=continuum_model,
            )
        )
        self.evaluation_stage = evaluation.build()
