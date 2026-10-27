# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Independent determinant-Hessian oracle, including cross-particle backflow."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

from jaqmc.app.moire.data import MoireData
from jaqmc.app.moire.optimized.gamma_directions import kinetic_grouped
from jaqmc.app.moire.optimized.gamma_operator import kinetic_analytic, potential


def fixture():
    rng = np.random.default_rng(292)
    momenta = rng.normal(size=(4, 2, 2))
    coef = rng.normal(size=(4, 2, 2, 2, 4)) + 1j * rng.normal(size=(4, 2, 2, 2, 4))

    class Wavefunction:
        def layer_components(self, params, data):
            k = jnp.asarray(momenta, dtype=data.positions.dtype)
            c = jnp.asarray(
                coef, dtype=jnp.result_type(data.positions.dtype, jnp.complex64)
            )
            waves = jnp.exp(1j * jnp.einsum("knd,nd->k", k, data.positions))
            phi = jnp.einsum("mdijk,k->mdij", c, waves)
            chi = jnp.exp(1j * data.spin_coords[:, None] * jnp.asarray([2, 1, -1, -2]))
            return phi, chi, jnp.einsum("il,ldij->dij", chi, phi)

    model = {
        "parameters": np.array([-16.0, -5.0, 0.043, -0.035, 0.012, -0.015]),
        "star": np.array(
            [[0.1, 0], [-0.05, np.sqrt(3) * 0.05], [-0.05, -np.sqrt(3) * 0.05]]
        ),
    }
    return (
        Wavefunction(),
        rng.normal(size=(2, 2)),
        np.array([0.21, 0.47]),
        np.array([1.0, -1.0]),
        model,
    )


def determinant_oracle(wf, r, s, signs, model, length):
    """Differentiate actual row-replaced determinants for an independent oracle."""
    phi, chi, base = wf.layer_components({}, MoireData(positions=r, spin_coords=s))
    reference = jnp.sum(jnp.linalg.det(base))
    sx = jnp.kron(jnp.eye(2), jnp.array([[0.0, 1.0], [1.0, 0.0]]))
    sy = jnp.kron(jnp.eye(2), jnp.array([[0.0, -1j], [1j, 0.0]]))
    ops = jnp.stack(
        [
            jnp.broadcast_to(jnp.eye(4), (2, 4, 4)),
            jnp.broadcast_to(sx, (2, 4, 4)),
            signs[:, None, None] * sy,
        ]
    )

    def numerator(x):
        p, c, b = wf.layer_components({}, MoireData(positions=x, spin_coords=s))
        changed = jnp.einsum("il,oilm,mdij->odij", c, ops, p)
        return jnp.stack(
            [
                jnp.stack(
                    [
                        jnp.sum(jnp.linalg.det(b.at[:, i, :].set(changed[o, :, i, :])))
                        / reference
                        for i in range(2)
                    ]
                )
                for o in range(3)
            ]
        )

    hessian = jax.jit(jax.jacfwd(jax.jacfwd(numerator)))(r)
    iso = sum(hessian[0, i, i, 0, i, 0] + hessian[0, i, i, 1, i, 1] for i in range(2))
    quad = sum(
        hessian[1, i, i, 0, i, 0]
        - hessian[1, i, i, 1, i, 1]
        + 2 * hessian[2, i, i, 0, i, 1]
        for i in range(2)
    )
    kinetic = (
        (model["parameters"][0] * iso + model["parameters"][1] * quad)
        * 1000
        / length**2
    )
    matrix = jax.vmap(lambda x, z: potential(x, z, model))(r * length, signs) * 1000
    changed = jnp.einsum("il,ilm,mdij->dij", chi, matrix, phi)
    return kinetic + sum(
        jnp.sum(jnp.linalg.det(base.at[:, i, :].set(changed[:, i, :]))) / reference
        for i in range(2)
    )


@pytest.mark.parametrize("method", ["baseline", "jvp", "lap", "sparse-lap"])
def test_matrix_energy_matches_determinant_hessian(method):
    wf, r, s, signs, model = fixture()
    with enable_x64():
        r, s, signs = map(jnp.asarray, (r, s, signs))
        expected = determinant_oracle(wf, r, s, signs, model, 66.771640395)
        if method == "baseline":
            value = kinetic_analytic(
                wf, {}, r, s, signs, model, 66.771640395, include_potential=True
            )
        else:
            value = kinetic_grouped(
                wf,
                {},
                r,
                s,
                signs,
                model,
                66.771640395,
                include_potential=True,
                method=method,
                directions=6,
            )
        np.testing.assert_allclose(value, expected, rtol=1e-9, atol=1e-9)
    with enable_x64(False):
        r, s, signs = [jnp.asarray(x, dtype=jnp.float32) for x in (r, s, signs)]
        fn = kinetic_analytic if method == "baseline" else kinetic_grouped
        kwargs = {} if method == "baseline" else {"method": method, "directions": 6}
        value = jax.jit(
            lambda x: fn(
                wf,
                {},
                x,
                s,
                signs,
                model,
                66.771640395,
                include_potential=True,
                **kwargs,
            )
        )(r)
        assert value.dtype == jnp.complex64
        np.testing.assert_allclose(value, expected, rtol=3e-4, atol=3e-3)


def test_invalid_direction_options():
    wf, r, s, signs, model = fixture()
    with pytest.raises(ValueError):
        kinetic_grouped(wf, {}, r, s, signs, model, 66.0, method="random")
    with pytest.raises(ValueError):
        kinetic_grouped(wf, {}, r, s, signs, model, 66.0, directions=0)
