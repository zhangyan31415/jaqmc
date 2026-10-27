# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Adapter wiring preserves Coulomb and estimator-local derivative choices."""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from jaqmc.app.moire.optimized import gamma
from jaqmc.estimator.base import PerWalkerEstimator


class DummyCoulomb(PerWalkerEstimator):
    def evaluate_single_walker(self, params, data, prev_walker_stats, state, rngs):
        return {"energy:coulomb": 0.0}, state


def stage_fixture():
    grad = SimpleNamespace(
        f_log_psi=lambda p, d: 0j,
        clip_method="median",
        clip_scale=5.0,
        loss_key="energy:total",
    )
    coulomb = DummyCoulomb(vmap_chunk_size=1)
    return SimpleNamespace(
        estimators=SimpleNamespace(estimators={"grads": grad, "coulomb": coulomb})
    ), coulomb


def test_install_keeps_methods_local_and_coulomb(monkeypatch):
    monkeypatch.setattr(
        gamma.kfac_utils, "set_use_cholesky_inversion", lambda enabled: None
    )
    wavefunction = SimpleNamespace(nlayers=4, nspins=(1, 1))
    first, coulomb = stage_fixture()
    gamma.install_backend(
        first, wavefunction, {}, 66.0, derivative_method="jvp", directions=6
    )
    second, _ = stage_fixture()
    gamma.install_backend(second, wavefunction, {}, 66.0, derivative_method="baseline")
    assert first.estimators.estimators["gamma"].derivative_method == "jvp"
    assert second.estimators.estimators["gamma"].derivative_method == "baseline"
    assert first.estimators.estimators["coulomb"] is coulomb
    assert coulomb.vmap_chunk_size == 512
    assert first.estimators.estimators["grads"].loss_key == "energy:total"
    assert first.estimators.estimators["gamma"].directions == 6
    assert jnp.array(first.estimators.estimators["gamma"].spin_signs).tolist() == [
        1.0,
        -1.0,
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"derivative_method": "other"},
        {"directions": 0},
        {"directions": 1.5},
        {"chunk": 0},
        {"length_A": -1.0},
    ],
)
def test_install_rejects_invalid_configuration(kwargs):
    stage, _ = stage_fixture()
    args = dict(
        stage=stage,
        wavefunction=SimpleNamespace(nlayers=4, nspins=(1, 1)),
        model={},
        length_A=66.0,
    )
    args.update(kwargs)
    with pytest.raises(ValueError):
        gamma.install_backend(**args)
