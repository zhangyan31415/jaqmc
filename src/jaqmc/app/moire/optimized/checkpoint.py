# Copyright (c) 2026 JaQMC contributors
# SPDX-License-Identifier: Apache-2.0

"""Native full-state NPZ checkpoints with explicit PRNG and update counters."""

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jaqmc.utils.checkpoint import tree_from_npz, tree_to_npz


def optimizer_counts(state, updates):
    arrays = tree_to_npz({"opt_state": state.opt_state})
    counts = {
        k: int(np.asarray(v))
        for k, v in arrays.items()
        if k.endswith("/count") and np.asarray(v).ndim == 0
    }
    if len(counts) < 2 or any(v != updates for v in counts.values()):
        raise ValueError(
            f"LR/KFAC counts differ from completed updates {updates}: {counts}"
        )
    return counts


def save_checkpoint(path, state, key, updates):
    optimizer_counts(state, updates)
    arrays = tree_to_npz(state)
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise FloatingPointError("Nonfinite full checkpoint state")
    rng = np.asarray(key)
    if rng.shape != (2,) or rng.dtype != np.uint32:
        raise ValueError("Checkpoint requires a legacy uint32 PRNG key of shape(2,)")
    arrays.update(step=np.asarray(updates - 1, dtype=np.int64), rng_key=rng)
    path = Path(path)
    temporary = path.with_suffix(".partial.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def load_checkpoint(path, reference):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: np.asarray(archive[k]) for k in archive.files}
    if "rng_key" not in arrays or "step" not in arrays:
        raise ValueError("Checkpoint requires explicit PRNG and global step")
    required = tree_to_npz(reference)
    if set(arrays) != set(required) | {"step", "rng_key"}:
        raise ValueError("Native checkpoint state schema differs")
    for name, expected in required.items():
        if (
            arrays[name].shape != np.shape(expected)
            or arrays[name].dtype != np.asarray(expected).dtype
        ):
            raise ValueError(f"Checkpoint shape/dtype mismatch: {name}")
        if not np.isfinite(arrays[name]).all():
            raise FloatingPointError(f"Nonfinite checkpoint array: {name}")
    rng, step = arrays["rng_key"], arrays["step"]
    if rng.shape != (2,) or rng.dtype != np.uint32:
        raise ValueError("Invalid checkpoint PRNG")
    if step.ndim != 0 or not np.issubdtype(step.dtype, np.integer) or int(step) < -1:
        raise ValueError("Invalid checkpoint step")
    updates = int(step) + 1
    state = tree_from_npz(arrays, reference)
    optimizer_counts(state, updates)
    return state, jnp.asarray(rng), updates
