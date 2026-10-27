# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Exact Gamma directional jets; optional grouping, no Hamiltonian changes."""

import jax
import jax.numpy as jnp
import numpy as np

from jaqmc.app.moire.hamiltonian import _orbitals_to_det_weights
from jaqmc.laplacian import (
    LapTuple,
    Local1Jacobian,
    OwnerRole,
    OwnerRoles,
    forward_laplacian,
)

from .gamma_operator import _contract, _direction, _operator_stack, potential


def kinetic_grouped(  # noqa: C901 -- exact derivative branches share one contraction
    wf,
    params,
    r,
    s,
    spin_signs,
    model,
    length_A,
    include_potential=False,
    *,
    method="lap",
    directions=3,
):
    if method not in ("jvp", "lap", "sparse-lap"):
        raise ValueError(method)
    if (
        not isinstance(directions, int)
        or isinstance(directions, bool)
        or directions < 1
    ):
        raise ValueError("direction group must be a positive integer")
    if method == "sparse-lap":
        from .folx_compat import install

        install()
    stack, phi, chi = _operator_stack(wf, params, r, s, spin_signs)
    weights, inverse = _orbitals_to_det_weights(stack[0])
    a1, a2 = model["parameters"][:2]

    def orbital_fn(p, x, angles, signs):
        return _operator_stack(wf, p, x, angles, signs)[0]

    transformed = forward_laplacian(orbital_fn) if method != "jvp" else None

    def contribution(raw_index):
        # Pad the last group with a valid index; discard its contribution below.
        index = jnp.minimum(raw_index, 3 * r.shape[0] - 1)
        i, axis, v = _direction(r, index)
        if method == "jvp":

            def f(x):
                return orbital_fn(params, x, s, spin_signs)

            def first_fn(x):
                return jax.jvp(f, (x,), (v,))[1]

            first, second = jax.jvp(first_fn, (r,), (v,))
        else:
            jac = v[None]
            if method == "sparse-lap":
                # One latent scalar t: r(t)=r+t*v. All input positions have
                # the same latent owner; zero weights still mean no dependence.
                jac = Local1Jacobian(
                    blocks=v[None, None],
                    owners=OwnerRoles(OwnerRole(None, np.array([0]))),
                    input_shape=(1, 1),
                    input_owner_axis=0,
                )
            result = transformed(
                params, LapTuple(r, jac, jnp.zeros_like(r)), s, spin_signs
            )
            first, second = result.dense_jacobian[0], result.laplacian
        bfirst = jnp.einsum("dij,djk->dik", inverse, first[0])
        bsecond = jnp.einsum("dij,djk->dik", inverse, second[0])
        logfirst = jnp.trace(bfirst, axis1=-2, axis2=-1)
        detsecond = (
            logfirst**2
            + jnp.trace(bsecond, axis1=-2, axis2=-1)
            - jnp.einsum("dij,dji->d", bfirst, bfirst)
        )
        col = inverse[:, :, i]
        cfirst = jnp.einsum("dij,dj->di", bfirst, col)
        csecond = jnp.einsum("dij,dj->di", bsecond, col)
        ctwice = jnp.einsum("dij,dj->di", bfirst, cfirst)
        row = stack[1:, :, i, :]
        drow = first[1:, :, i, :]
        ddrow = second[1:, :, i, :]
        t = jnp.einsum("odj,dj->od", row, col)
        dt = jnp.einsum("odj,dj->od", drow, col) - jnp.einsum("odj,dj->od", row, cfirst)
        ddt = (
            jnp.einsum("odj,dj->od", ddrow, col)
            - 2 * jnp.einsum("odj,dj->od", drow, cfirst)
            + 2 * jnp.einsum("odj,dj->od", row, ctwice)
            - jnp.einsum("odj,dj->od", row, csecond)
        )
        mixed_second = t * detsecond[None] + 2 * logfirst[None] * dt + ddt
        derivatives = jnp.concatenate(
            [
                jnp.sum(weights * detsecond)[None],
                jnp.einsum("od,d->o", mixed_second, weights),
            ]
        )
        return _contract(derivatives, axis, a1, a2)

    def body(total, group):
        ids = group * directions + jnp.arange(directions)
        values = jax.vmap(contribution)(ids)
        updated = jax.lax.fori_loop(
            0,
            directions,
            lambda k, acc: acc + jnp.where(ids[k] < 3 * r.shape[0], values[k], 0),
            total,
        )
        return updated, None

    count = (3 * r.shape[0] + directions - 1) // directions
    result = (
        jax.lax.scan(
            body,
            jnp.asarray(0j, dtype=jnp.result_type(stack, a1, a2)),
            jnp.arange(count),
        )[0]
        * 1000
        / length_A**2
    )
    if include_potential:
        mat = (
            jax.vmap(lambda x, z: potential(x, z, model))(r * length_A, spin_signs)
            * 1000
        )
        changed = jnp.einsum("il,ilm,mdij->dij", chi, mat, phi)
        result = result + jnp.einsum("d,dij,dji->", weights, changed, inverse)
    return result
