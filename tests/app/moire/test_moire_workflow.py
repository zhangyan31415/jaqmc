# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.app.moire.matrix_hamiltonian import (
    MatrixContinuumDifferential,
    MatrixContinuumObservable,
    MatrixContinuumPotential,
)
from jaqmc.app.moire.model import MatrixContinuumModel
from jaqmc.app.moire.workflow import (
    MoireJointMCMCSampler,
    MoireTrainWorkflow,
    configure_system,
    load_continuum_model,
    make_estimators,
)
from jaqmc.estimator.structure_factor import PeriodicStructureFactor
from jaqmc.estimator.total_energy import TotalEnergy
from jaqmc.utils.config import ConfigManager
from jaqmc.writer.console import FieldSpec


def _base_config(tmp_path: Path) -> dict:
    return {
        "workflow": {"batch_size": 4, "save_path": str(tmp_path / "run")},
        "system": {
            "electron_spins": [4, 0],
            "nelec": 4,
            "supercell_matrix": [[2, 0], [0, 2]],
        },
        "wf": {"hidden_dims_single": [8], "hidden_dims_double": [4], "ndets": 1},
    }


def test_saved_config_excludes_runtime_klist_arrays(tmp_path: Path):
    cfg = _base_config(tmp_path)
    manager = ConfigManager(cfg)
    configure_system(manager)

    yaml = manager.to_yaml()

    runtime_fields = {
        "nspins",
        "primitive_lattice",
        "simulation_lattice",
        "twist",
        "phase_mode",
        "singlephase_klists",
        "multiphase_klist",
        "translation_vectors",
    }
    assert runtime_fields.isdisjoint(manager.resolved_config["wf"])
    assert "singlephase_klists" not in yaml
    assert "multiphase_klist" not in yaml
    assert "translation_vectors" not in yaml


def test_fractional_spin_channels_use_multiphase_at_integer_filling(tmp_path: Path):
    cfg = _base_config(tmp_path)
    cfg["system"]["electron_spins"] = [2, 2]

    system, wf = configure_system(ConfigManager(cfg))

    assert system.scale == 4
    assert wf.phase_mode == "multiphase"
    assert wf.multiphase_klist is not None
    assert wf.multiphase_klist.shape == (4, 2)
    assert wf.translation_vectors is not None


def test_continuum_model_path_configures_generic_workflow(tmp_path: Path):
    model = MatrixContinuumModel(
        momentum_powers=np.asarray([[0, 0], [2, 0]]),
        momentum_matrices=np.asarray(
            [
                np.diag([0.1, 0.2, 0.3]),
                np.diag([1.0, 1.5, 2.0]),
            ]
        ),
        fourier_vectors=np.asarray([[0.1, 0.0]]),
        fourier_matrices=np.asarray(
            [[[0.01, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.03]]]
        ),
        basis_labels=("a", "b", "c"),
        energy_unit="eV",
        length_unit="angstrom",
    )
    model_path = tmp_path / "model.npz"
    model.save(model_path)
    cfg = _base_config(tmp_path)
    cfg["continuum_model"] = {"path": str(model_path)}
    cfg["estimators"] = {"enabled": {"coulomb": False}}
    manager = ConfigManager(cfg)

    system, wf = configure_system(manager)
    converted = load_continuum_model(manager, system)
    assert converted is not None
    estimators = make_estimators(
        manager,
        wf,
        system,
        continuum_model=converted,
    )

    assert wf.nlayers == 3
    assert converted.energy_unit == "meV"
    assert converted.length_unit == "dimensionless"
    assert set(estimators) == {
        "matrix_potential",
        "matrix_differential",
        "total",
    }
    assert isinstance(estimators["matrix_potential"], MatrixContinuumPotential)
    assert isinstance(estimators["matrix_differential"], MatrixContinuumDifferential)
    assert "kinetic" not in estimators
    assert "soc" not in estimators
    assert "moire_potential" not in estimators


def test_observable_collections_are_wired_from_config(tmp_path: Path):
    cfg = _base_config(tmp_path)
    cfg.pop("workflow")
    cfg["estimators"] = {
        "enabled": {"energy": False},
        "internal_observables": {
            "polarizations": {
                "module": (
                    "jaqmc.app.moire.matrix_hamiltonian:MatrixContinuumObservable"
                ),
                "operator_matrices_real": [
                    [[1.0, 0.0], [0.0, -1.0]],
                    [[0.0, 1.0], [1.0, 0.0]],
                ],
                "output_keys": [
                    "observable:internal_z",
                    "observable:internal_x",
                ],
            }
        },
        "periodic_observables": {
            "charge_order": {
                "module": "jaqmc.estimator.structure_factor:PeriodicStructureFactor",
                "reciprocal_indices": [[1, 0], [0, 1], [1, 1]],
            }
        },
    }
    manager = ConfigManager(cfg)
    system, wf = configure_system(manager)

    estimators = make_estimators(manager, wf, system)

    assert set(estimators) == {
        "internal_observable:polarizations",
        "periodic_observable:charge_order",
    }
    observable = estimators["internal_observable:polarizations"]
    assert isinstance(observable, MatrixContinuumObservable)
    assert observable.f_internal_components == wf.internal_components
    structure_factor = estimators["periodic_observable:charge_order"]
    assert isinstance(structure_factor, PeriodicStructureFactor)
    np.testing.assert_allclose(structure_factor.lattice, system.supercell_lattice)
    assert structure_factor.data_field == "positions"
    manager.finalize()


def test_joint_proposal_moves_positions_and_spin_coords_together():
    lattice = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    sampler = MoireJointMCMCSampler(spin_mass=2.0)
    sampler.configure(lattice)
    proposal = sampler.sampling_proposal
    data = {"positions": jnp.zeros((32, 2, 2)), "spin_coords": jnp.zeros((32, 2))}
    out = proposal(jax.random.PRNGKey(0), data, jnp.array(0.2))
    assert out["positions"].shape == data["positions"].shape
    assert out["spin_coords"].shape == data["spin_coords"].shape
    pos_step = np.asarray(out["positions"] - data["positions"])
    spin_step = np.asarray((out["spin_coords"] + np.pi) % (2.0 * np.pi) - np.pi)
    frac_positions = np.asarray(out["positions"] @ jnp.linalg.inv(lattice))
    assert np.std(pos_step) > 0.0
    assert np.std(spin_step) > 0.0
    assert np.all(frac_positions >= 0.0)
    assert np.all(frac_positions < 1.0)
    assert np.all(np.asarray(out["spin_coords"]) >= 0.0)
    assert np.all(np.asarray(out["spin_coords"]) < 2.0 * np.pi)
    np.testing.assert_allclose(
        np.std(spin_step),
        2.0 * np.pi * 0.2 / 2.0,
        rtol=0.25,
    )


def test_default_console_fields_exist_in_complex_total_energy_reduce():
    # Moire energies are complex (the SOC term carries a 1j factor), so
    # TotalEnergy.reduce takes the complex branch and only produces
    # ``total_energy_real`` / ``total_energy_real_var`` (never ``total_energy_var``).
    # Guard the default console preset against referencing a missing key.
    preset = MoireTrainWorkflow.default_preset()
    console_fields = preset["train"]["writers"]["console"]["fields"]
    referenced_keys = {
        FieldSpec.parse(field.strip()).key for field in console_fields.split(",")
    }

    total = TotalEnergy(components=["energy:kinetic"])
    walker_stats = {"total_energy": jnp.asarray([1.0 + 0.5j, 2.0 - 0.3j, 1.5 + 0.1j])}
    reduced = total.reduce(walker_stats)

    energy_keys = referenced_keys - {"pmove"}
    missing = energy_keys - set(reduced)
    assert not missing, (
        f"console preset references stat keys absent from reduce(): {missing}. "
        f"Available: {sorted(reduced)}"
    )
