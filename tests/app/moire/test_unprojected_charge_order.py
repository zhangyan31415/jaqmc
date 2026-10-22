# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Physical boundary contracts for supercell charge-order wavefunctions."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.moire.config import MoireConfig
from jaqmc.app.moire.data import MoireData
from jaqmc.app.moire.wavefunction import MoireWavefunction
from jaqmc.app.moire.workflow import configure_system
from jaqmc.utils.config import ConfigManager


def _wavefunction(*, projected=False, twist=(0.17, -0.23)):
    config = MoireConfig(
        nelec=2,
        electron_spins=(2, 0),
        supercell_matrix=[[2, 0], [0, 2]],
        twist=list(twist),
    )
    wf = MoireWavefunction(
        nspins=config.electron_spins,
        primitive_lattice=jnp.asarray(config.lattice_vectors),
        simulation_lattice=jnp.asarray(config.supercell_lattice),
        twist=jnp.asarray(twist),
        hidden_dims_single=[8],
        hidden_dims_double=[4],
        ndets=2,
        momentum_projection=projected,
    )
    wf.configure_phase_inputs(config)
    rng = jax.random.PRNGKey(91)
    key_pos, key_spin, key_params = jax.random.split(rng, 3)
    data = MoireData(
        positions=jax.random.uniform(key_pos, (2, 2))
        @ jnp.asarray(config.supercell_lattice),
        spin_coords=jax.random.uniform(key_spin, (2,), maxval=2 * jnp.pi),
    )
    return wf, config, data, wf.init_params(data, key_params)


@pytest.mark.requires_x64
def test_unprojected_wavefunction_obeys_single_particle_twisted_boundary():
    wf, config, data, params = _wavefunction()
    original = jnp.exp(wf.logpsi(params, data))
    for axis in range(2):
        positions = data.positions.at[0].add(
            jnp.asarray(config.supercell_lattice)[axis]
        )
        moved = data.merge({"positions": positions})
        np.testing.assert_allclose(
            jnp.exp(wf.logpsi(params, moved)) / original,
            jnp.exp(2j * jnp.pi * config.twist[axis]),
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.requires_x64
def test_unprojected_wavefunction_is_antisymmetric_and_not_primitive_projected():
    wf, config, data, params = _wavefunction(twist=(0.0, 0.0))
    original = jnp.exp(wf.logpsi(params, data))
    swapped = data.merge(
        {
            "positions": data.positions[::-1],
            "spin_coords": data.spin_coords[::-1],
        }
    )
    np.testing.assert_allclose(
        jnp.exp(wf.logpsi(params, swapped)) / original,
        -1.0,
        rtol=1e-6,
        atol=1e-6,
    )
    translated = data.merge(
        {"positions": data.positions + jnp.asarray(config.lattice_vectors)[0]}
    )
    assert not np.isclose(
        abs(jnp.exp(wf.logpsi(params, translated)) / original), 1.0, atol=1e-4
    )
    assert wf.orbitals(params, data).shape == (2, 2, 2)
    assert wf.k_com is None
    assert wf.translation_vectors is None


def test_config_exposes_unprojected_wavefunction_at_fractional_and_integer_filling():
    for count in (2, 4):
        cfg = ConfigManager(
            {
                "system": {
                    "nelec": count,
                    "electron_spins": [count, 0],
                    "supercell_matrix": [[2, 0], [0, 2]],
                },
                "wf": {"momentum_projection": False},
            }
        )
        _, wf = configure_system(cfg)
        assert wf.phase_mode == "multiphase"
        assert wf.k_com is None
        assert wf.multiphase_klist.shape == (4, 2)
        assert not cfg.resolved_config["wf"]["momentum_projection"]


def test_unprojected_configuration_rejects_conflicting_momentum_sector():
    with pytest.raises(ValueError, match="k_com must be None"):
        configure_system(
            ConfigManager(
                {
                    "wf": {
                        "momentum_projection": False,
                        "k_com": [0.0, 0.0],
                    }
                }
            )
        )
