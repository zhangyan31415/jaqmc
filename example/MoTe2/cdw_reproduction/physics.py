"""Trainable CDW Slater/FermiNet ansatz and native full-continuum energy."""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "true")
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from jaqmc.app.moire.data import MoireData
from jaqmc.app.moire.hamiltonian import MoirePotential, MoireSOC
from jaqmc.app.moire.wavefunction import MoireWavefunction
from jaqmc.estimator.ewald import EwaldSum2D
from jaqmc.estimator.kinetic import EuclideanKinetic, LaplacianMode
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.output.logdet import LogDet
from reference import configuration, exact_slater, spectral_seed

jax.config.update("jax_enable_x64", True)


def microbatch_map(function, r, s, size):
    chunk = min(size, r.shape[0])
    if r.shape[0] % chunk:
        raise ValueError("Walker count must divide microbatch size.")
    shape = (r.shape[0]//chunk, chunk)
    result = jax.lax.map(lambda x: jax.vmap(function)(*x),
                        (r.reshape(*shape, *r.shape[1:]),
                         s.reshape(*shape, *s.shape[1:])))
    return jax.tree.map(lambda x: x.reshape(r.shape[0], *x.shape[2:]), result)


class CDWWavefunction(MoireWavefunction):
    """Supercell Fourier orbital coefficients plus an unrestricted neural core.

    The seed is represented in parameters, so restoring the checkpoint needs
    only the basis vectors. No motif, pinning field, centers, or fixed Gaussian
    survives in the forward pass. Boundary twist is applied exactly once.
    """
    spectral_g: jnp.ndarray = runtime_dep()
    initial_coeff: jnp.ndarray = runtime_dep()
    residual_scale: float = .02

    @nn.compact
    def _get_unprojected_multiphase_orbitals(self, data):
        coeff_r = self.param("fourier_real",
            lambda key: jnp.asarray(self.initial_coeff).real)
        coeff_i = self.param("fourier_imag",
            lambda key: jnp.asarray(self.initial_coeff).imag)
        plane = jnp.exp(1j*data.positions@jnp.asarray(self.spectral_g).T)
        base = jnp.einsum("ig,lgj->lij", plane, coeff_r+1j*coeff_i)
        residual = self._get_orbital_blocks(data)
        return tuple(self._apply_twist(base[layer][None, ...] +
            self.residual_scale*residual[layer], data.positions)
            for layer in range(self.nlayers))

    def _get_orbital_blocks(self, data):
        embedding = self.feature_layer(data.positions)
        # Squared chord distances avoid the norm's nonsmooth zero argument.
        one = embedding["ae_features"].at[:, 0].set(
            embedding["ae_features"][:, 0]**2)
        pair = embedding["ee_features"].at[:, :, 0].set(
            embedding["ee_features"][:, :, 0]**2)
        hidden, _ = self.backbone_layer(one, pair)
        return tuple(jnp.transpose(real(hidden)+1j*imag(hidden), (1, 0, 2))
            for real, imag in zip(self.real_orbital_layers, self.imag_orbital_layers))


class Model:
    def __init__(self, theta=3., filling=1, cells=12, motif="MX",
                 walkers=32, width=32, depth=3, ndets=1, cutoff=9,
                 kinetic_mode="forward_laplacian", coulomb=True,
                 free_reference=False):
        self.cfg = configuration(theta, filling, cells)
        self.n = self.cfg.nelec
        self.walkers = walkers
        self.cell = jnp.asarray(self.cfg.supercell_lattice)
        self.invcell = jnp.linalg.inv(self.cell)
        self.pref = self.cfg.kinetic_prefactor_mev/self.cfg.effective_mass
        g, coeff, self.centers = spectral_seed(self.cfg, motif, cutoff=cutoff)
        self.exact_energy = None
        if free_reference:
            if coulomb:
                raise ValueError("The exact free reference requires Coulomb off.")
            g, coeff, self.exact_energy, _ = exact_slater(self.cfg, cutoff)
        self.wf = CDWWavefunction(nspins=self.cfg.electron_spins,
            primitive_lattice=jnp.asarray(self.cfg.lattice_vectors),
            simulation_lattice=self.cell, twist=jnp.asarray(self.cfg.twist),
            momentum_projection=False, spectral_g=jnp.asarray(g),
            initial_coeff=jnp.asarray(coeff), hidden_dims_single=[width]*depth,
            hidden_dims_double=[min(16, width)]*depth, ndets=ndets,
            residual_scale=0. if free_reference else .02)
        self.wf.configure_phase_inputs(self.cfg)
        self.key = jax.random.PRNGKey(20261022)
        self.kine = EuclideanKinetic(f_log_psi=self.logdata,
            data_field="positions", prefactor=self.pref, mode=LaplacianMode(kinetic_mode))
        primitive = jnp.asarray(self.cfg.lattice_vectors)
        self.soc = MoireSOC(f_layer_components=self.components,
            primitive_lattice=primitive, nspins=self.cfg.electron_spins,
            prefactor=self.pref)
        self.pot = MoirePotential(f_layer_components=self.components,
            primitive_lattice=primitive, nspins=self.cfg.electron_spins,
            v1_mev=self.cfg.v1_mev, phi1_rad=self.cfg.phi1_rad,
            omega1_mev=self.cfg.omega1_mev)
        self.ewald = EwaldSum2D(self.cell)
        self.coulomb = coulomb
        self.logs = jax.jit(jax.vmap(self.log, in_axes=(None, 0, 0)))
        self.energy_batch = jax.jit(lambda p, r, s:
            microbatch_map(lambda x, y: self.energy(p, x, y), r, s, 4))

    def jastrow(self, r):
        i, j = np.triu_indices(self.n, 1)
        theta = 2*jnp.pi*((r[i]-r[j])@self.invcell)
        a = jnp.sin(theta)@self.cell/(2*jnp.pi)
        b = (1-jnp.cos(theta))@self.cell/(2*jnp.pi)
        distance = jnp.sqrt(jnp.sum(a*a+b*b, axis=-1)+1e-24)
        slope = self.cfg.coulomb_prefactor_mev/self.pref if self.coulomb else 0.
        return -jnp.sum(slope*.3/(1+distance/.3))

    def components(self, p, data):
        phi, chi, base = self.wf.layer_components(p, data)
        factor = jnp.exp(self.jastrow(data.positions)/self.n)
        return phi*factor, chi, base*factor

    def logdata(self, p, data):
        return self.wf.logpsi(p, data) + self.jastrow(data.positions)

    def log(self, p, r, s):
        return self.logdata(p, MoireData(positions=r, spin_coords=s))

    def energy(self, p, r, s):
        data = MoireData(positions=r, spin_coords=s)
        k = self.kine.evaluate_single_walker(p, data, {}, None, self.key)[0]["energy:kinetic"]
        soc = self.soc.evaluate_single_walker(p, data, {}, None, self.key)[0]["energy:soc"]
        v = self.pot.evaluate_single_walker(p, data, {}, None, self.key)[0]["energy:moire_potential"]
        c = (self.ewald.energy(r, -jnp.ones(self.n)) *
             self.cfg.coulomb_prefactor_mev if self.coulomb else 0.)
        return jnp.array([k+soc+v+c, k, soc, v, c])/self.n

    def initialize(self, seed):
        a, b, c = jax.random.split(jax.random.PRNGKey(seed), 3)
        noise = .18*jax.random.normal(a, (self.walkers, self.n, 2), dtype=jnp.float64)
        r = jnp.mod((jnp.asarray(self.centers)[None, ...]+noise)@self.invcell, 1.)@self.cell
        s = jax.random.uniform(b, (self.walkers, self.n), dtype=jnp.float64)*2*jnp.pi
        p = self.wf.init_params(MoireData(positions=r[0], spin_coords=s[0]), c)
        p = jax.tree.map(lambda x: x.astype(jnp.float64), p)
        return p, r, s

    def _position_candidate(self, r, width, normal_key, kernel_key, spatial_kernel):
        noise = jax.random.normal(normal_key, r.shape)
        if spatial_kernel == "full":
            increment = width*noise
        else:
            mode_key, index_key, translation_key = jax.random.split(kernel_key, 3)
            mode = jax.random.uniform(mode_key, (self.walkers,))
            index = jax.random.randint(index_key, (self.walkers,), 0, self.n)
            local = width*jnp.sqrt(self.n)*noise*jax.nn.one_hot(index, self.n)[:, :, None]
            directions = jnp.asarray([[1, 0], [-1, 0], [0, 1], [0, -1]])
            choice = jax.random.randint(translation_key, (self.walkers,), 0, 4)
            translation = directions[choice] @ jnp.asarray(self.cfg.lattice_vectors)
            increment = jnp.where((mode < .1)[:, None, None], translation[:, None, :],
                jnp.where((mode < .4)[:, None, None], local, width*noise))
        return jnp.mod((r+increment) @ self.invcell, 1.) @ self.cell

    def make_sampler(self, sweeps=20, cache_orbitals=True, spatial_kernel="full"):
        """Metropolis sweeps, caching position-only orbitals during angle moves.

        The cache lives within one call so parameter updates cannot stale it.
        ``cache_orbitals=False`` retains the original evaluation path.
        ``spatial_kernel="mixture"`` uses 60% full Gaussian, 30% single-particle
        Gaussian, and 10% rigid +/- primitive translations. Each proposal is
        symmetric on the periodic cell and uses the unchanged MH target.
        """
        if spatial_kernel not in {"full", "mixture"}:
            raise ValueError("Spatial kernel must be full or mixture.")
        if not cache_orbitals:
            return self._make_uncached_sampler(sweeps, spatial_kernel)

        def orbitals(p, r, s):
            return self.wf.layer_components(
                p, MoireData(positions=r, spin_coords=s))[0]

        def from_orbitals(phi, s, jastrow):
            chi = jnp.stack([jnp.exp(1j*s), jnp.exp(-1j*s)], axis=-1)
            matrix = jnp.einsum("il,ldij->dij", chi, phi)
            return LogDet().apply({}, matrix)["logpsi"] + jastrow

        orbital_batch = jax.vmap(orbitals, in_axes=(None, 0, 0))
        jastrow_batch = jax.vmap(self.jastrow)
        log_batch = jax.vmap(from_orbitals)

        def run(p, state, width):
            r, s, key = state
            phi = orbital_batch(p, r, s)
            jastrow = jastrow_batch(r)
            logs = log_batch(phi, s, jastrow)

            def step(carry, _):
                r, s, phi, jastrow, logs, key, acceptance = carry
                key, a, b, c, d = jax.random.split(key, 5)
                candidate = self._position_candidate(r, width, a, d, spatial_kernel)
                proposed_phi = orbital_batch(p, candidate, s)
                proposed_jastrow = jastrow_batch(candidate)
                proposed = log_batch(proposed_phi, s, proposed_jastrow)
                take = jnp.log(jax.random.uniform(b, (self.walkers,))) < 2*(proposed.real-logs.real)
                r = jnp.where(take[:, None, None], candidate, r)
                phi = jnp.where(take[:, None, None, None, None], proposed_phi, phi)
                jastrow = jnp.where(take, proposed_jastrow, jastrow)
                logs = jnp.where(take, proposed, logs)

                def angle_step(index, carry):
                    s, logs, spin_key, count = carry
                    spin_key, proposal_key, accept_key = jax.random.split(spin_key, 3)
                    angles = s.at[:, index].set(
                        jax.random.uniform(proposal_key, (self.walkers,))*2*jnp.pi)
                    proposed = log_batch(phi, angles, jastrow)
                    spin_take = jnp.log(jax.random.uniform(accept_key, (self.walkers,))) < 2*(proposed.real-logs.real)
                    return (jnp.where(spin_take[:, None], angles, s),
                            jnp.where(spin_take, proposed, logs),
                            spin_key, count+spin_take.mean())

                s, logs, _, spin_count = jax.lax.fori_loop(
                    0, self.n, angle_step, (s, logs, c, jnp.array(0.)))
                acceptance += jnp.array([take.mean(), spin_count/self.n])
                return (r, s, phi, jastrow, logs, key, acceptance), None

            result, _ = jax.lax.scan(step,
                (r, s, phi, jastrow, logs, key, jnp.zeros(2)), None, length=sweeps)
            return (result[0], result[1], result[5]), result[6]/sweeps
        return jax.jit(run)

    def _make_uncached_sampler(self, sweeps, spatial_kernel):
        """Full position and independent uniform-angle Metropolis sweeps."""
        def run(p, state, width):
            r, s, key = state
            logs = self.logs(p, r, s)
            def step(carry, _):
                r, s, logs, key, acceptance = carry
                key, a, b, c, d = jax.random.split(key, 5)
                candidate = self._position_candidate(r, width, a, d, spatial_kernel)
                proposed = self.logs(p, candidate, s)
                take = jnp.log(jax.random.uniform(b, (self.walkers,))) < 2*(proposed.real-logs.real)
                r = jnp.where(take[:, None, None], candidate, r)
                logs = jnp.where(take, proposed, logs)
                def angle_step(index, carry):
                    s, logs, spin_key, count = carry
                    spin_key, proposal_key, accept_key = jax.random.split(spin_key, 3)
                    angles = s.at[:, index].set(
                        jax.random.uniform(proposal_key, (self.walkers,))*2*jnp.pi)
                    proposed = self.logs(p, r, angles)
                    spin_take = jnp.log(jax.random.uniform(accept_key, (self.walkers,))) < 2*(proposed.real-logs.real)
                    return (jnp.where(spin_take[:, None], angles, s),
                            jnp.where(spin_take, proposed, logs),
                            spin_key, count+spin_take.mean())
                s, logs, _, spin_count = jax.lax.fori_loop(
                    0, self.n, angle_step, (s, logs, c, jnp.array(0.)))
                acceptance += jnp.array([take.mean(), spin_count/self.n])
                return (r, s, logs, key, acceptance), None
            result, _ = jax.lax.scan(step, (r, s, logs, key, jnp.zeros(2)), None, length=sweeps)
            return (result[0], result[1], result[3]), result[4]/sweeps
        return jax.jit(run)
