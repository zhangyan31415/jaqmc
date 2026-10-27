# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Full four-component Gamma hole Hamiltonian and exact backflow derivatives.

The four internal components are layer x molecular orbital, not four layers.
Coordinates are simulation units; ``length_A`` converts them to Angstrom.
Hamiltonian coefficients are eV Angstrom squared (a1/a2) and eV (potential),
and the returned local energy is meV per many-particle configuration.
"""

import jax
import jax.numpy as jnp


def internal_matrices(spin):
    eye = jnp.eye(2)
    sx = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    sy = jnp.array([[0.0, -1j], [1j, 0.0]]) * spin
    sz = jnp.diag(jnp.array([1.0, -1.0]))
    return (
        jnp.kron(eye, sx),
        jnp.kron(eye, sy),
        jnp.kron(eye, sz),
        jnp.kron(sx, eye),
        sz,
    )


def potential(positions, spin, model, holes=True):
    _, _, v1, v2, w1, w2 = model["parameters"]
    eye = jnp.eye(2)
    sx = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    sy = jnp.array([[0.0, -1j], [1j, 0.0]]) * spin
    sz = jnp.diag(jnp.array([1.0, -1.0]))
    c = jnp.cos(positions @ jnp.asarray(model["star"]).T)
    fx = jnp.sqrt(3.0) * (c[..., 1] - c[..., 2])
    fy = 2 * c[..., 0] - c[..., 1] - c[..., 2]
    v = v1 * jnp.kron(eye, sz) + v2 * (
        fx[..., None, None] * jnp.kron(sz, sx) + fy[..., None, None] * jnp.kron(sz, sy)
    )
    v = v + (w1 + 2 * w2 * jnp.sum(c, axis=-1))[..., None, None] * jnp.kron(sx, eye)
    return -v if holes else v


def quadrupole_second(numerator, positions):
    """Sum (dxx-dyy)N_x,i + 2*dxy N_y,i, including all backflow.

    numerator returns the unnormalized O_i Psi amplitudes, divided by a
    spatially CONSTANT reference amplitude. Differentiating O_i Psi/Psi(r)
    directly without restoring Psi would produce an incorrect kinetic energy.

    Returns:
        The complex summed quadrupole derivative of the supplied numerator.
    """
    n = positions.shape[0]

    # Nested directional JVPs avoid a full Jacobian/Hessian and retain all
    # dependence on other electron rows. Three directions per particle suffice.
    def body(total, index):
        i = index // 3
        direction = index % 3
        ex = jnp.zeros_like(positions).at[i, 0].set(1.0)
        ey = jnp.zeros_like(positions).at[i, 1].set(1.0)
        tangent = jnp.where(direction == 0, ex, jnp.where(direction == 1, ey, ex + ey))

        def first(x):
            return jax.jvp(numerator, (x,), (tangent,))[1]

        d2 = jax.jvp(first, (positions,), (tangent,))[1][:, i]
        # 2*dxy=(dx+dy)^2-dxx-dyy.
        term = jnp.where(
            direction == 0,
            d2[0] - d2[1],
            jnp.where(direction == 1, -d2[0] - d2[1], d2[1]),
        )
        return total + term, None

    return jax.lax.scan(
        body,
        jnp.asarray(0j, dtype=jnp.result_type(numerator(positions))),
        jnp.arange(3 * n),
    )[0]


def potential_local(wf, params, r, s, spin_sign, model, length_A):
    from jaqmc.app.moire.data import MoireData
    from jaqmc.app.moire.hamiltonian import compute_operator_ratio

    phi, chi, base = wf.layer_components(params, MoireData(positions=r, spin_coords=s))
    mat = (
        jax.vmap(lambda x, z: potential(x, z, model))(r * length_A, spin_sign) * 1000.0
    )
    transformed = jnp.einsum("il,ilm,mdij->dij", chi, mat, phi)
    return jnp.sum(compute_operator_ratio(base, transformed))


def kinetic_quadrupole(wf, params, r, s, spin_sign, model, length_A):
    from jaqmc.app.moire.data import MoireData
    from jaqmc.app.moire.hamiltonian import compute_operator_ratio

    reference = jax.lax.stop_gradient(
        wf.logpsi(params, MoireData(positions=r, spin_coords=s))
    )
    sx = jnp.kron(jnp.eye(2), jnp.array([[0.0, 1.0], [1.0, 0.0]]))
    sy = jnp.kron(jnp.eye(2), jnp.array([[0.0, -1j], [1j, 0.0]]))
    matrices = jnp.stack(
        [jnp.broadcast_to(sx, (r.shape[0], 4, 4)), spin_sign[:, None, None] * sy]
    )

    def numerator(x):
        d = MoireData(positions=x, spin_coords=s)
        phi, chi, base = wf.layer_components(params, d)
        transformed = jnp.einsum("il,oilm,mdij->odij", chi, matrices, phi)
        ratio = compute_operator_ratio(base, transformed)
        return jnp.exp(wf.logpsi(params, d) - reference) * ratio

    return (
        model["parameters"][1] * 1000.0 / length_A**2 * quadrupole_second(numerator, r)
    )


def _operator_stack(wf, params, r, s, signs):
    from jaqmc.app.moire.data import MoireData

    phi, chi, base = wf.layer_components(params, MoireData(positions=r, spin_coords=s))
    sx = jnp.kron(jnp.eye(2), jnp.array([[0.0, 1.0], [1.0, 0.0]]))
    sy = jnp.kron(jnp.eye(2), jnp.array([[0.0, -1j], [1j, 0.0]]))
    ops = jnp.stack(
        [jnp.broadcast_to(sx, (r.shape[0], 4, 4)), signs[:, None, None] * sy]
    )
    mixed = jnp.einsum("il,oilm,mdij->odij", chi, ops, phi)
    return jnp.concatenate([base[None], mixed], axis=0), phi, chi


def _direction(r, index):
    i = index // 3
    axis = index % 3
    ex = jnp.zeros_like(r).at[i, 0].set(1.0)
    ey = jnp.zeros_like(r).at[i, 1].set(1.0)
    return i, axis, jnp.where(axis == 0, ex, jnp.where(axis == 1, ey, ex + ey))


def _contract(second, axis, a1, a2):
    # 2*dxy = d_(x+y)^2 - d_x^2 - d_y^2.
    return jnp.where(
        axis == 0,
        a1 * second[0] + a2 * (second[1] - second[2]),
        jnp.where(
            axis == 1, a1 * second[0] + a2 * (-second[1] - second[2]), a2 * second[2]
        ),
    )


def kinetic_fused(wf, params, r, s, signs, model, length_A):
    """Correctness bridge: fuse scalar and matrix numerator derivatives.

    Returns:
        The many-particle kinetic local energy in meV.
    """
    from jaqmc.app.moire.data import MoireData
    from jaqmc.app.moire.hamiltonian import compute_operator_ratio

    reference = jax.lax.stop_gradient(
        wf.logpsi(params, MoireData(positions=r, spin_coords=s))
    )

    def numerator(x):
        orbitals, _, _ = _operator_stack(wf, params, x, s, signs)
        ratios = compute_operator_ratio(orbitals[0], orbitals[1:])
        ratio = jnp.concatenate([jnp.ones_like(ratios[:1]), ratios], axis=0)
        amplitude = jnp.exp(
            wf.logpsi(params, MoireData(positions=x, spin_coords=s)) - reference
        )
        return amplitude * ratio

    a1, a2 = model["parameters"][:2]

    def body(total, index):
        i, axis, v = _direction(r, index)

        def first(x):
            return jax.jvp(numerator, (x,), (v,))[1]

        second = jax.jvp(first, (r,), (v,))[1][:, i]
        return total + _contract(second, axis, a1, a2), None

    total = jax.lax.scan(
        body,
        jnp.array(0.0 + 0.0j, dtype=jnp.result_type(numerator(r), a1, a2)),
        jnp.arange(3 * r.shape[0]),
    )[0]
    return total * 1000 / length_A**2


def kinetic_analytic(wf, params, r, s, signs, model, length_A, include_potential=False):
    """Second derivatives of F*t, F=det(A), t=R_i A^{-1}_{:,i}.

    L=tr(A^-1 A'), F''/F=L²+tr(A^-1 A''-(A^-1 A')²).
    All orbital rows are differentiated, including rows other than i. Operator
    amplitudes may be zero: t is never used as a divisor. Multideterminant
    weights are the actual signed complex det(A_d)/sum_d det(A_d).

    Returns:
        The kinetic local energy in meV, with optional matrix potential.
    """
    from jaqmc.app.moire.hamiltonian import _orbitals_to_det_weights

    def orbitals(x):
        return _operator_stack(wf, params, x, s, signs)[0]

    stack, phi, chi = _operator_stack(wf, params, r, s, signs)
    weights, inverse = _orbitals_to_det_weights(stack[0])
    a1, a2 = model["parameters"][:2]

    def body(total, index):
        i, axis, v = _direction(r, index)

        def first_fn(x):
            return jax.jvp(orbitals, (x,), (v,))[1]

        first, second = jax.jvp(first_fn, (r,), (v,))
        # Base logdet derivatives, shared by both orbital operators.
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
        return total + _contract(derivatives, axis, a1, a2), None

    value = (
        jax.lax.scan(
            body,
            jnp.array(0.0 + 0.0j, dtype=jnp.result_type(stack, a1, a2)),
            jnp.arange(3 * r.shape[0]),
        )[0]
        * 1000
        / length_A**2
    )
    if include_potential:
        mat = jax.vmap(lambda x, z: potential(x, z, model))(r * length_A, signs) * 1000
        transformed = jnp.einsum("il,ilm,mdij->dij", chi, mat, phi)
        value = value + jnp.einsum("d,dij,dji->", weights, transformed, inverse)
    return value
