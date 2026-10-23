"""Independent Hamiltonian mapping and motif/boundary/kinetic contracts."""
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from jax import numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[3] /
                      "example/MoTe2/cdw_reproduction"))
from reference import configuration, electron_fields, hole_model, motif_centers
from jaqmc.app.moire.hamiltonian import _moire_potential_matrix


@pytest.mark.requires_x64
def test_source_particle_hole_mapping_matches_native_real_space():
    cfg = configuration(3., 1, 12)
    r = np.random.default_rng(42).normal(size=(31, 2))
    native = _moire_potential_matrix(jnp.asarray(r),
        jnp.asarray(cfg.lattice_vectors), jnp.ones(len(r)),
        v1_mev=cfg.v1_mev, phi1_rad=cfg.phi1_rad, omega1_mev=cfg.omega1_mev)
    source_hole = -electron_fields(r).swapaxes(-1, -2)
    np.testing.assert_allclose(native, source_hole, atol=1e-11)
    model = hole_model(cfg)
    b = model.constant_matrix
    fourier = sum(matrix[None, ...]*np.exp(1j*r@vector)[:, None, None] +
                  matrix.conj().T[None, ...]*np.exp(-1j*r@vector)[:, None, None]
                  for vector, matrix in zip(model.fourier_vectors, model.fourier_matrices))
    offset = np.diag(np.diag(b))
    np.testing.assert_allclose(b[None, ...]+fourier-offset, source_hole, atol=1e-11)


@pytest.mark.parametrize("cells", [3, 12, 27])
@pytest.mark.parametrize("filling", [1, 2])
def test_cdw_geometry(cells, filling):
    cfg = configuration(4.2, filling, cells)
    for motif in ["MM", "MX", "XM"]:
        centers = motif_centers(cfg, motif)
        assert centers.shape == (cfg.nelec, 2)
        fraction = centers@np.linalg.inv(cfg.supercell_lattice)
        assert np.all(fraction >= -1e-10) and np.all(fraction < 1+1e-10)
        assert len(np.unique(np.round(fraction, 8), axis=0)) == cfg.nelec
    # All three primitive K harmonics are representable on this supercell.
    q = np.array([[1/3, 1/3], [2/3, -1/3], [-1/3, 2/3]])
    np.testing.assert_allclose(q@np.asarray(cfg.supercell_matrix).T,
        np.round(q@np.asarray(cfg.supercell_matrix).T), atol=1e-10)


@pytest.mark.requires_x64
def test_trainable_seed_boundaries_and_kinetic_parity():
    from physics import Model
    from jaqmc.app.moire.data import MoireData
    scan = Model(cells=3, filling=2, width=4, depth=1, walkers=4,
                 cutoff=2, kinetic_mode="scan")
    forward = Model(cells=3, filling=2, width=4, depth=1, walkers=4,
                    cutoff=2, kinetic_mode="forward_laplacian")
    p, r, s = scan.initialize(42)
    assert all(x.dtype == jnp.float64 for x in jax.tree.leaves(p))
    x = MoireData(positions=r[0], spin_coords=s[0])
    z = jnp.exp(scan.logdata(p, x))
    for vector in scan.cell:
        moved = x.merge({"positions": x.positions.at[0].add(vector)})
        np.testing.assert_allclose(jnp.exp(scan.logdata(p, moved))/z, 1., atol=1e-8)
    swapped = x.merge({"positions": x.positions[::-1],
                       "spin_coords": x.spin_coords[::-1]})
    np.testing.assert_allclose(jnp.exp(scan.logdata(p, swapped))/z, -1., atol=1e-8)
    first = jax.jit(scan.energy)(p, r[0], s[0])
    second = jax.jit(forward.energy)(p, r[0], s[0])
    np.testing.assert_allclose(first, second, rtol=1e-8, atol=1e-8)
    assert np.isfinite(first).all()


@pytest.mark.requires_x64
@pytest.mark.parametrize("filling", [1, 2])
def test_frozen_energy_batch_size_preserves_all_complex_components(filling):
    from physics import Model, microbatch_map
    model = Model(cells=3, filling=filling, width=4, depth=1,
                  walkers=8, cutoff=2)
    params, r, s = model.initialize(125)
    reference = np.asarray(model.energy_batch(params, r, s))
    larger = jax.jit(lambda p, a, b: microbatch_map(
        lambda x, y: model.energy(p, x, y), a, b, 8))
    measured = np.asarray(larger(params, r, s))
    assert np.isfinite(measured).all()
    np.testing.assert_allclose(measured, reference, rtol=1e-8, atol=1e-7)
    np.testing.assert_allclose(measured[:, 0], measured[:, 1:].sum(axis=1),
                               rtol=1e-8, atol=1e-7)


@pytest.mark.requires_x64
@pytest.mark.parametrize("filling", [1, 2])
@pytest.mark.parametrize("ndets", [1, 2])
@pytest.mark.parametrize("kernel", ["full", "mixture"])
def test_cached_sampler_preserves_trajectory_across_parameter_updates(filling, ndets, kernel):
    from physics import Model
    m = Model(cells=27, filling=filling, walkers=4, width=8, depth=2,
              ndets=ndets, cutoff=3)
    p, r, s = m.initialize(42)
    old = m.make_sampler(sweeps=3, cache_orbitals=False, spatial_kernel=kernel)
    cached = m.make_sampler(sweeps=3, spatial_kernel=kernel)
    state = r, s, jax.random.PRNGKey(5100)
    for width in [.07, .2]:
        first, acceptance = old(p, state, width)
        second, cached_acceptance = cached(p, state, width)
        for expected, actual in zip(first, second):
            np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(acceptance, cached_acceptance)
        np.testing.assert_allclose(m.logs(p, first[0], first[1]),
                                   m.logs(p, second[0], second[1]), atol=1e-12)
        state = first
        p = jax.tree.map(lambda x: x*1.001, p)
