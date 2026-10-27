# Copyright (c) 2026 JaQMC contributors
# SPDX-License-Identifier: Apache-2.0

"""Process-local sparse reduction compatibility for JAX 0.11 explicit sharding.

Original source is unchanged. Native low-level reduce_sum_p.bind calls omit
out_sharding for derived Jacobian blocks; public jnp.sum supplies the required
parameter. Restrict replacement to owner-local sparse sum; dense path native.
"""

import jax
import jax.numpy as jnp

from jaqmc.laplacian.guards import is_sparse_laptuple
from jaqmc.laplacian.primitives.reductions import handle_reduce_sum as original
from jaqmc.laplacian.primitives.registry import (
    ensure_register_builtins,
    register_builtin_handler,
)
from jaqmc.laplacian.primitives.sparse_ops import sparse_output_block_axes
from jaqmc.laplacian.types import LapTuple


def handle_sparse_reduce_sum(args, kwargs):
    x = args[0]
    if not is_sparse_laptuple(x):
        return original(args, kwargs)
    axes = kwargs["axes"]
    axes = tuple(range(x.x.ndim)) if axes is None else tuple(a % x.x.ndim for a in axes)
    y = jnp.sum(x.x, axis=axes)
    lap = jnp.sum(x.laplacian, axis=axes)
    owners = x.jacobian.owners.reduce_output_axes(
        axes, output_shape=x.jacobian.output_shape
    )
    if owners is not None:
        blocks = jnp.sum(
            x.jacobian.blocks, axis=sparse_output_block_axes(axes, output_ndim=x.x.ndim)
        )
        jac = x.jacobian.with_blocks(blocks, owners=owners)
    else:
        jac = jnp.sum(x.jacobian.to_dense(), axis=tuple(a + 1 for a in axes))
    return LapTuple(y, jac, lap)


def install():
    ensure_register_builtins()
    register_builtin_handler(jax.lax.reduce_sum_p, handle_sparse_reduce_sum)
