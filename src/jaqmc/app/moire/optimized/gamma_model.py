# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Load a frozen six-parameter Gamma model without an external model runtime."""

import json
from pathlib import Path

import numpy as np


def load_model(path):
    """Return parameters and geometry in eV/Angstrom units.

    The JSON schema stores a fixed physical model, not executable Python. ``q``
    and ``rotation`` are included for reproducibility; the real-space operator
    uses the three-vector reciprocal star in the model coordinate frame.

    Raises:
        ValueError: If the model schema, shapes, values, or geometry are invalid.
    """
    path = Path(path).expanduser().resolve()
    raw = json.loads(path.read_text())
    if raw.get("schema_version") != 1:
        raise ValueError("Gamma model requires schema_version=1.")
    expected = {"parameters": (6,), "first": (2,), "star": (3, 2), "rotation": (2, 2)}
    model = {}
    for name, shape in expected.items():
        try:
            array = np.asarray(raw[name], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing or invalid model field: {name}") from exc
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"{name} must have shape {shape} and finite real values.")
        model[name] = array
    try:
        q = np.asarray(raw["q"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Missing or invalid reciprocal q points.") from exc
    if q.ndim != 2 or q.shape[1] != 2 or q.shape[0] == 0 or not np.isfinite(q).all():
        raise ValueError("q must be a nonempty, finite (n, 2) real array.")
    if np.linalg.norm(model["first"]) <= 0:
        raise ValueError("The first reciprocal vector must be nonzero.")
    if not np.allclose(
        model["rotation"] @ model["rotation"].T, np.eye(2), atol=1e-12, rtol=0
    ):
        raise ValueError("rotation must be orthogonal.")
    angles = np.arange(3) * 2 * np.pi / 3
    first = model["first"]
    star = np.array(
        [
            np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ first
            for a in angles
        ]
    )
    if not np.allclose(model["star"], star, atol=1e-12, rtol=1e-10):
        raise ValueError(
            "star must contain the first reciprocal vector and its C3 rotations."
        )
    model.update(
        q=q,
        path=str(path),
        length_A=float(4 * np.pi / (np.sqrt(3) * np.linalg.norm(first))),
    )
    return model
