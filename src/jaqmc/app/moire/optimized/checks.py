# Copyright (c) 2026 JaQMC contributors
# SPDX-License-Identifier: Apache-2.0

"""Raw finite checks before native NaN-aware reductions."""

import jax
import jax.numpy as jnp
import numpy as np

from jaqmc.estimator.base import Estimator


class RawFiniteEstimator(Estimator):
    def __init__(self, inner, name):
        self.inner = inner
        self.key = "raw_nonfinite_count:" + name

    def init(self, data, rngs):
        return self.inner.init(data, rngs)

    def evaluate_batch_walkers(
        self, params, batched_data, prev_walker_stats, state, rngs
    ):
        return self.inner.evaluate_batch_walkers(
            params, batched_data, prev_walker_stats, state, rngs
        )

    def reduce(self, walker_stats):
        count = sum(jnp.sum(~jnp.isfinite(v)) for v in jax.tree.leaves(walker_stats))
        return {**self.inner.reduce(walker_stats), self.key: count}

    def finalize_stats(self, batched_stats, state):
        ordinary = {k: v for k, v in batched_stats.items() if k != self.key}
        return {
            **self.inner.finalize_stats(ordinary, state),
            self.key: jnp.max(batched_stats[self.key]),
        }

    def finalize_state(self, state, *, n_steps):
        return self.inner.finalize_state(state, n_steps=n_steps)


def check_stats(stats):
    if not all(np.isfinite(v).all() for v in jax.tree.leaves(stats)):
        raise FloatingPointError("Nonfinite training statistics")
    count = sum(
        int(np.asarray(v).sum()) for k, v in stats.items() if "nonfinite_count" in k
    )
    residual = max(
        (
            float(np.asarray(v))
            for k, v in stats.items()
            if k.endswith("spin_inverse_residual_max")
        ),
        default=0.0,
    )
    if count or residual >= 1e-3:
        raise FloatingPointError(
            f"Raw numerical failure: count={count}, inverse residual={residual}"
        )
    return count, residual
