# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Portable Gamma model validation; no external model runtime is imported."""

import json
from pathlib import Path

import numpy as np
import pytest

from jaqmc.app.moire.optimized.gamma_model import load_model

MODEL = (
    Path(__file__).resolve().parents[4]
    / "example/moire_optimized/models/zrs2_gamma_3p15.json"
)


def test_frozen_model_has_physical_geometry():
    model = load_model(MODEL)
    assert model["parameters"].shape == (6,)
    assert model["star"].shape == (3, 2)
    assert model["q"].ndim == 2 and model["q"].shape[1] == 2
    assert model["length_A"] == pytest.approx(66.771640395, rel=1e-9)
    assert "runtime" not in model
    np.testing.assert_allclose(
        model["rotation"] @ model["rotation"].T, np.eye(2), atol=1e-12
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("parameters", [1, 2]),
        ("first", [0, 0]),
        ("rotation", [[1, 1], [0, 1]]),
        ("q", [[0, float("nan")]]),
    ],
)
def test_malformed_model_rejected(tmp_path, key, value):
    raw = json.loads(MODEL.read_text())
    raw[key] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_model(path)
