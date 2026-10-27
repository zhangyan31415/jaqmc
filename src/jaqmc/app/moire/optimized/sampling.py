# Copyright (c) 2026 JaQMC contributors
# SPDX-License-Identifier: Apache-2.0

"""Exact binary-layer conditional Gibbs composed with position Metropolis.

Opt-in replacement preserving the native MCMC state layout. Layer labels occupy
spin_coords as float 0/1;
physical electron-spin sectors are unchanged. Singular/nonfinite algebra is
flagged, never silently dropped or denominator-clipped. All recorded raw flags
must be zero and the post-refactor inverse residual must be < 1e-3.
"""

from dataclasses import replace
from types import MethodType

import jax
import jax.numpy as jnp

from jaqmc.geometry.pbc import wrap_positions
from jaqmc.sampler.mcmc import MCMCState
from jaqmc.utils import parallel_jax


def initialize_labels(batched_data, key):
    """Fresh Bernoulli layer labels; preserve source positions, then burn in.

    Never round continuous angles. This initializes a new representation and
    does not assert equilibrium. Call once before a discrete chain's burn-in.

    Returns:
        New batched data with unchanged positions and Bernoulli layer labels.
    """
    old = batched_data.data.spin_coords
    labels = jax.random.bernoulli(key, 0.5, old.shape).astype(old.dtype)
    return replace(batched_data, data=batched_data.data.merge({"spin_coords": labels}))


def selected_matrix(phi, labels):
    """Select each electron's layer row from phi of shape (B,2,1,N,N).

    Returns:
        The selected orbital matrices with shape (B,N,N).
    """
    return jnp.where(labels[:, :, None] == 0, phi[:, 0, 0], phi[:, 1, 0])


def conditional_probability(phi, inverse, i):
    """Compute a stably rescaled binary conditional from determinant ratios.

    Returns:
        Layer-zero probabilities and a validity mask that callers must check.
    """
    column = inverse[:, :, i]
    a = jnp.sum(phi[:, 0, 0, i, :] * column, axis=-1)
    b = jnp.sum(phi[:, 1, 0, i, :] * column, axis=-1)
    scale = jnp.maximum(jnp.abs(a), jnp.abs(b))
    valid = jnp.isfinite(a) & jnp.isfinite(b) & jnp.isfinite(scale) & (scale > 0)
    safe_scale = jnp.where(valid, scale, 1.0)
    # Finite placeholder ONLY for invalid chains. Consumers must fail the run.
    a = jnp.where(valid, a / safe_scale, 1.0 + 0j)
    b = jnp.where(valid, b / safe_scale, 0.0 + 0j)
    wa = jnp.abs(a) ** 2
    wb = jnp.abs(b) ** 2
    return wa / (wa + wb), valid


def gibbs_sweep(phi, labels, key):
    """Perform a full sequential heat-bath sweep invariant under |det Phi_L|^2.

    Returns:
        Updated binary labels and numerical validity/residual diagnostics.

    Raises:
        ValueError: If phi is not a bilayer single-determinant orbital tensor.
    """
    if phi.ndim != 5 or phi.shape[1:3] != (2, 1):
        raise ValueError("Binary Gibbs requires phi(B,2,1,N,N)")
    matrix = selected_matrix(phi, labels)
    inverse = jnp.linalg.inv(matrix)
    size = labels.shape[-1]
    identity = jnp.eye(size)
    error = jnp.max(jnp.abs(matrix @ inverse - identity))
    bad = (
        jnp.sum(~jnp.isfinite(phi))
        + jnp.sum(~jnp.isfinite(labels))
        + jnp.sum((labels != 0) & (labels != 1))
    )

    def update(i, carry):
        labels, matrix, inverse, key, bad, residual_max, raw_max, refactors, changed = (
            carry
        )
        inverse = jax.lax.cond(
            i % 4 == 0, lambda _: jnp.linalg.inv(matrix), lambda _: inverse, None
        )
        residual_max = jnp.maximum(
            residual_max, jnp.max(jnp.abs(matrix @ inverse - identity))
        )
        p0, valid = conditional_probability(phi, inverse, i)
        bad += jnp.sum(~valid)
        key, draw_key = jax.random.split(key)
        value = (jax.random.uniform(draw_key, p0.shape) >= p0).astype(labels.dtype)
        column = inverse[:, :, i]
        new_row = jnp.where(value[:, None] == 0, phi[:, 0, 0, i, :], phi[:, 1, 0, i, :])
        delta = new_row - matrix[:, i, :]
        denominator = jnp.sum(new_row * column, axis=-1)
        inverse = (
            inverse
            - column[:, :, None]
            * jnp.einsum("bi,bij->bj", delta, inverse)[:, None, :]
            / denominator[:, None, None]
        )
        matrix = matrix.at[:, i, :].set(new_row)
        raw_error = jnp.max(jnp.abs(matrix @ inverse - identity), axis=(-2, -1))
        raw_max = jnp.maximum(raw_max, jnp.max(raw_error))
        refactor = (
            (~jnp.isfinite(inverse).all(axis=(-2, -1)))
            | (jnp.abs(denominator) < 1e-5)
            | (~jnp.isfinite(raw_error))
            | (raw_error > 1e-3)
        )
        inverse = jax.lax.cond(
            jnp.any(refactor), lambda _: jnp.linalg.inv(matrix), lambda _: inverse, None
        )
        corrected = jnp.max(jnp.abs(matrix @ inverse - identity))
        residual_max = jnp.maximum(residual_max, corrected)
        refactors += jnp.any(refactor).astype(jnp.int32)
        changed += jnp.sum(value != labels[:, i])
        return (
            labels.at[:, i].set(value),
            matrix,
            inverse,
            key,
            bad,
            residual_max,
            raw_max,
            refactors,
            changed,
        )

    labels, matrix, inverse, key, bad, residual_max, raw_max, refactors, changed = (
        jax.lax.fori_loop(
            0,
            size,
            update,
            (
                labels,
                matrix,
                inverse,
                key,
                bad,
                error,
                error,
                jnp.array(0),
                jnp.array(0),
            ),
        )
    )
    bad += (
        jnp.sum(~jnp.isfinite(labels))
        + jnp.sum(~jnp.isfinite(inverse))
        + jnp.sum(~jnp.isfinite(matrix))
    )
    return labels, {
        "spin_nonfinite_count": bad,
        "spin_inverse_residual_max": residual_max,
        "spin_inverse_residual_before_refactor_max": raw_max,
        "spin_corrective_refactor_count": refactors,
        "layer_changed_count": changed,
        "layer_update_count": jnp.array(labels.size),
    }


def install(stage, wavefunction, position_steps=20, layer_sweeps=1, freeze_width=False):
    """Install before binary burn, after onehot wavefunction patch.

    Rebuild orbitals once at entry for current parameters; reuse only within
    this draw. Position proposals are symmetric wrapped Cartesian moves.
    Width adaptation uses position acceptance only. Freeze for measurements.

    Raises:
        ValueError: If the model, step counts or native sampler plan is unsupported.
        RuntimeError: If the discrete representation was not selected first.
    """
    if (
        wavefunction.nlayers != 2
        or wavefunction.ndets != 1
        or wavefunction.phase_mode != "singlephase"
    ):
        raise ValueError("Requires bilayer singlephase one determinant")
    from jaqmc.app.moire.wavefunction import MoireWavefunction

    from .discrete import _discrete_components

    if MoireWavefunction.get_layer_components is not _discrete_components:
        raise RuntimeError(
            "Call prepare_representation('discrete') before workflow construction."
        )
    if position_steps < 1 or layer_sweeps < 1:
        raise ValueError("Positive step/sweep counts required")
    plan = stage.sample_plan
    if tuple(plan.samplers) != (("positions", "spin_coords"),):
        raise ValueError("Requires exactly one positions/spin_coords sampler")
    name = "positions,spin_coords"
    sampler = plan.samplers["positions", "spin_coords"]
    lattice = jnp.asarray(wavefunction.simulation_lattice)

    def step(self, params, batched_data, state, rngs):
        data = batched_data.data
        batch_layers = jax.vmap(
            lambda d: wavefunction.layer_components(params, d)[0],
            (batched_data.vmap_axis,),
        )
        phi = batch_layers(data)
        labels = data.spin_coords
        logprob = 2 * jnp.linalg.slogdet(selected_matrix(phi, labels))[1]
        rngs, position_key, layer_key = jax.random.split(rngs, 3)
        old_state = state[name]
        bad = jnp.sum(~jnp.isfinite(phi)) + jnp.sum(~jnp.isfinite(logprob))

        def position_move(_, carry):
            positions, phi, logprob, key, accepted, bad = carry
            key, proposal_key, accept_key = jax.random.split(key, 3)
            candidate = wrap_positions(
                positions
                + old_state.stddev
                * jax.random.normal(proposal_key, positions.shape)
                @ lattice,
                lattice,
            )
            proposed_phi = batch_layers(data.merge({"positions": candidate}))
            proposed_logprob = (
                2 * jnp.linalg.slogdet(selected_matrix(proposed_phi, labels))[1]
            )
            # A proposed zero-density node (-inf) is valid and gets rejected.
            bad += jnp.sum(~jnp.isfinite(proposed_phi)) + jnp.sum(
                jnp.isnan(proposed_logprob) | jnp.isposinf(proposed_logprob)
            )
            accept = proposed_logprob - logprob > jnp.log(
                jax.random.uniform(accept_key, logprob.shape)
            )
            return (
                jnp.where(accept[:, None, None], candidate, positions),
                jnp.where(accept[:, None, None, None, None], proposed_phi, phi),
                jnp.where(accept, proposed_logprob, logprob),
                key,
                accepted + jnp.sum(accept),
                bad,
            )

        positions, phi, logprob, _, accepted, bad = jax.lax.fori_loop(
            0,
            position_steps,
            position_move,
            (data.positions, phi, logprob, position_key, jnp.array(0), bad),
        )
        stats = {}
        for _ in range(layer_sweeps):
            layer_key, subkey = jax.random.split(layer_key)
            labels, sweep_stats = gibbs_sweep(phi, labels, subkey)
            for k, v in sweep_stats.items():
                stats[k] = (
                    jnp.maximum(stats.get(k, 0.0), v)
                    if k.endswith("_max")
                    else stats.get(k, 0) + v
                )
        pmove = parallel_jax.pmean(accepted / (position_steps * labels.shape[0]))
        stddev, pmoves, counter = old_state
        counter += 1
        index = counter % sampler.adapt_frequency
        pmoves = pmoves.at[index].set(pmove)
        if not freeze_width:
            stddev = jnp.where(
                index == 0,
                jnp.where(
                    jnp.mean(pmoves) > sampler.pmove_range[1],
                    stddev * 1.1,
                    jnp.where(
                        jnp.mean(pmoves) < sampler.pmove_range[0], stddev / 1.1, stddev
                    ),
                ),
                stddev,
            )
        stats["pmove"] = pmove
        stats["proposal_stddev"] = stddev
        stats["sampling_nonfinite_count"] = bad + jnp.sum(~jnp.isfinite(positions))
        state = {**state, name: MCMCState(stddev, pmoves, counter)}
        return (
            replace(
                batched_data,
                data=data.merge({"positions": positions, "spin_coords": labels}),
            ),
            stats,
            state,
        )

    plan.step = MethodType(step, plan)
