# Copyright (c) 2026 JaQMC contributors
# SPDX-License-Identifier: Apache-2.0

"""Portable runner configuration and complete-state continuation contracts."""

from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaqmc.app.moire.optimized.checkpoint import load_checkpoint, save_checkpoint
from jaqmc.app.moire.optimized.checks import RawFiniteEstimator, check_stats
from jaqmc.app.moire.optimized.runner import (
    build_config,
    config_fingerprint,
    create_parser,
)


def test_preset_paths_are_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for model in ("mote2", "gamma"):
        args = create_parser().parse_args(["--model", model, "--out", "run"])
        cfg = build_config(args)
        assert cfg["system"]["nelec"] == 18
        assert cfg["wf"]["nlayers"] == (2 if model == "mote2" else 4)
        assert cfg["workflow"]["save_path"] == str(tmp_path / "run")
        assert cfg["workflow"]["batch_size"] == 512
        if model == "gamma":
            assert "energy" not in cfg["estimators"]


def test_resume_fingerprint_ignores_output_but_preserves_physics():
    parser = create_parser()
    a = build_config(parser.parse_args(["--out", "a"]))
    b = build_config(parser.parse_args(["--out", "b"]))
    assert config_fingerprint(a) == config_fingerprint(b)
    b["system"]["dielectric_constant"] *= 2
    assert config_fingerprint(a) != config_fingerprint(b)


def test_invalid_representation_model_and_derivative_are_rejected():
    parser = create_parser()
    with pytest.raises(ValueError, match="discrete"):
        build_config(
            parser.parse_args(
                ["--model", "gamma", "--representation", "discrete", "--out", "x"]
            )
        )
    with pytest.raises(ValueError, match="derivative"):
        build_config(
            parser.parse_args(["--model", "mote2", "--derivative", "jvp", "--out", "x"])
        )


Counter = namedtuple("Counter", "count")
Optimizer = namedtuple("Optimizer", "optax kfac")
State = namedtuple("State", "params opt_state batched_data")


def test_checkpoint_restores_every_leaf_and_rng(tmp_path):
    state = State(
        {"w": jnp.array([1.0, 2.0])},
        Optimizer(Counter(jnp.array(17)), Counter(jnp.array(17))),
        {"r": jnp.ones((2, 2))},
    )
    key = jax.random.PRNGKey(731)
    path = tmp_path / "native.npz"
    save_checkpoint(path, state, key, 17)
    restored, restored_key, updates = load_checkpoint(path, state)
    assert updates == 17
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(key, restored_key)
    with np.load(path) as arrays:
        assert int(arrays["step"]) == 16
        assert arrays["rng_key"].dtype == np.uint32


def test_checkpoint_rejects_counter_reset_and_missing_rng(tmp_path):
    state = State(
        {"w": jnp.ones(1)}, Optimizer(Counter(jnp.array(0)), Counter(jnp.array(0))), {}
    )
    with pytest.raises(ValueError, match="count"):
        save_checkpoint(tmp_path / "bad.npz", state, jax.random.PRNGKey(1), 17)
    path = Path(tmp_path / "missing.npz")
    np.savez(path, step=0)
    with pytest.raises(ValueError, match="PRNG"):
        load_checkpoint(path, state)


def test_raw_nan_is_not_hidden_by_nan_aware_reduction():
    inner = SimpleNamespace(reduce=lambda x: {"loss": jnp.nanmean(x["raw"])})
    estimator = RawFiniteEstimator(inner, "fixture")
    reduced = estimator.reduce({"raw": jnp.array([1.0, jnp.nan])})
    assert float(reduced["loss"]) == pytest.approx(1.0)
    with pytest.raises(FloatingPointError, match="count=1"):
        check_stats(reduced)
