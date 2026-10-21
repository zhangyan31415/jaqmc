# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for primitives introduced/changed by JAX 0.11."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaqmc.laplacian import forward_laplacian, make_laplacian_input

pytestmark = pytest.mark.requires_x64


def test_jitted_gather_and_stack_matches_complex_hessian():
    def fn(x):
        rows = x[jnp.array([2, 0, 2])]
        joined = jnp.stack([rows, jnp.sin(rows)], axis=1)
        return jnp.sum(jnp.exp((.2 + .3j) * joined))
    x = jnp.arange(8.0, dtype=jnp.float64).reshape(4, 2) / 9
    result = jax.jit(
        lambda v: forward_laplacian(fn)(make_laplacian_input(v, sparse_axis=0))
    )(x)
    grad = jax.jacfwd(fn)(x)
    hess = jax.jacfwd(jax.jacfwd(fn))(x).reshape(x.size, x.size)
    np.testing.assert_allclose(result.dense_jacobian.reshape(x.shape), grad, atol=1e-11)
    np.testing.assert_allclose(result.laplacian, jnp.trace(hess), atol=1e-11)


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("sparse_axis", [None, 0])
def test_stack_linear_rule_with_constant(axis, sparse_axis, caplog):
    def fn(x):
        return jnp.stack([x, 2 * x, jnp.ones_like(x)], axis=axis)

    x = jnp.arange(6.0, dtype=jnp.float64).reshape(3, 2)
    with caplog.at_level("WARNING"):
        result = forward_laplacian(fn)(
            make_laplacian_input(x, sparse_axis=sparse_axis)
        )
    expected = jax.jacfwd(fn)(x).reshape(*fn(x).shape, x.size)
    np.testing.assert_allclose(jnp.moveaxis(result.dense_jacobian, 0, -1), expected)
    np.testing.assert_array_equal(result.laplacian, jnp.zeros_like(fn(x)))
    assert not any("Full Hessian fallback" in rec.message for rec in caplog.records)
