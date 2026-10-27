# Copyright (c) 2026 JaQMC contributors
# SPDX-License-Identifier: Apache-2.0

"""Centered VMC covariance and its equivalent weighted reverse pass.

Both preserve the native clipping rule. Weighted differentiation supports real
network parameters and complex log wavefunctions without materializing the
walker-by-parameter Jacobian.
"""

import jax
from jax import numpy as jnp

from jaqmc.estimator.loss_grad import LossAndGrad
from jaqmc.utils import parallel_jax
from jaqmc.utils.clip import clip_observable


class CenteredLossAndGrad(LossAndGrad):
    def reduce(self, walker_stats):
        energy = clip_observable(
            walker_stats["loss"], self.clip_method, scale=self.clip_scale
        )
        residual = energy - parallel_jax.pnanmean(energy)

        def covariance(g):
            centered = g - parallel_jax.pnanmean(g)
            shape = (residual.shape[0],) + (1,) * (centered.ndim - 1)
            return 2 * jnp.real(
                parallel_jax.pnanmean(jnp.conj(centered) * residual.reshape(shape))
            )

        return {
            "loss": parallel_jax.pnanmean(walker_stats["loss"]),
            "grads": jax.tree.map(covariance, walker_stats["grad_logpsi"]),
        }

    def finalize_stats(self, batched_stats, state):
        del state
        return jax.tree.map(lambda x: jnp.mean(x, axis=0), batched_stats)


class WeightedLossAndGrad(CenteredLossAndGrad):
    def evaluate_batch_walkers(
        self, params, batched_data, prev_walker_stats, state, rngs
    ):
        del rngs
        if any(jnp.iscomplexobj(p) for p in jax.tree.leaves(params)):
            raise ValueError(
                "Weighted scalar gradient currently requires real network parameters."
            )
        energy = prev_walker_stats[self.loss_key]
        clipped = clip_observable(energy, self.clip_method, scale=self.clip_scale)
        residual = clipped - parallel_jax.pnanmean(clipped)
        # Subtract the residual mean too: this equals centering the Jacobian
        # without materializing it, including finite precision residual centering.
        weights = jax.lax.stop_gradient(residual - parallel_jax.pnanmean(residual))
        valid = ~jnp.isnan(weights)
        weights = jnp.where(valid, weights, 0)
        denominator = parallel_jax.pmean(jnp.sum(valid))
        batch_logpsi = jax.vmap(self.f_log_psi, in_axes=(None, batched_data.vmap_axis))

        def objective(p):
            logs = batch_logpsi(p, batched_data.data)
            return 2 * jnp.sum(jnp.real(jnp.conj(logs) * weights)) / denominator

        grads = jax.grad(objective)(parallel_jax.pvary(params))
        grads = jax.tree.map(parallel_jax.pmean, grads)
        return {"loss": parallel_jax.pnanmean(energy), "grads": grads}, state

    def reduce(self, walker_stats):
        return walker_stats
