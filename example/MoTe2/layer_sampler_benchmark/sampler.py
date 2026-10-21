"""Exact-target kernels. Full determinant fallback reference for all proposals."""
import itertools
from nqs18 import Model,N,coefficients,log_contract,jax,jnp,np


def cluster_accept_log(delta,proxy_delta):
    return delta-proxy_delta


class Sampler(Model):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.chunks={}

    def initial(self, p, seed, continuous):
        k1, k2, key = jax.random.split(jax.random.PRNGKey(seed), 3)
        r = jax.random.uniform(k1, (self.walkers, N, 2), dtype=jnp.float64)@self.cell
        u = jax.random.uniform(k2, (self.walkers, N), dtype=jnp.float64)*2*jnp.pi if continuous else jax.random.randint(k2, (self.walkers, N), 0, 2)
        return (r, u, self.phi_batch(p, r), key)

    def make_chunk(self, method, sweeps=10, proxy_radius=.65):
        cache_key=(method,sweeps,proxy_radius)
        if cache_key in self.chunks:
            return self.chunks[cache_key]
        cont = method == 'continuous'
        W = self.walkers
        coeff_batch = jax.vmap(lambda u: coefficients(u, cont))
        batch_log = jax.vmap(log_contract)
        ids = jnp.arange(W)
        images = jnp.asarray(list(itertools.product([-1, 0, 1], repeat=2)))@self.cell

        def graph(r, coupling):
            # Nearest periodic image, then fixed positive surrogate couplings.
            delta = r[:, None, :]-r[None, :, :]
            frac = delta@self.invcell
            delta = (frac-jnp.round(frac))@self.cell
            d2 = jnp.min(jnp.sum((delta[:, :, None, :]+images)**2, axis=-1), axis=-1)
            return coupling*jnp.exp(-d2/(2*proxy_radius**2))*(1-jnp.eye(N))

        def cluster(u, J, key):
            kb, ks = jax.random.split(key)
            random = jax.random.uniform(kb, (N, N))
            bonds = jnp.triu((random < -jnp.expm1(-2*J)) & (u[:, None] == u[None, :]), 1)
            bonds = bonds | bonds.T
            seed = jax.random.randint(ks, (), 0, N)
            def grow(_, mask):
                return mask | jnp.any(bonds & mask[:, None], axis=0)
            mask = jax.lax.fori_loop(0, N, grow, jnp.arange(N) == seed)
            prop = jnp.where(mask, 1-u, u)
            z, zp = 1-2*u, 1-2*prop
            delta = .5*(zp@J@zp-z@J@z)
            return prop, delta, jnp.sum(mask)

        def sweep(p, carry, rw, sw, coupling):
            r, u, phi, key = carry
            key, kr, ka = jax.random.split(key, 3)
            rp = jnp.mod((r+rw*jax.random.normal(kr, r.shape))@self.invcell, 1.)@self.cell
            phip = self.phi_batch(p, rp)
            chi = coeff_batch(u)
            old = batch_log(phi, chi)
            new = batch_log(phip, chi)
            accept = jnp.log(jax.random.uniform(ka, (W,))) < 2*(new.real-old.real)
            r = jnp.where(accept[:, None, None], rp, r)
            phi = jnp.where(accept[:, None, None, None, None], phip, phi)
            logs = jnp.where(accept, new, old)
            J = jax.vmap(graph, in_axes=(0, None))(r, coupling) if method == 'wolff_corrected' else jnp.zeros((W, N, N))

            def spin(_, internal):
                u, logs, key, changes, sizes = internal
                key, ki, kp, ka = jax.random.split(key, 4)
                i = jax.random.randint(ki, (W,), 0, N)
                proxy_delta = jnp.zeros(W)
                size = jnp.ones(W)
                if method == 'wolff_corrected':
                    prop, proxy_delta, size = jax.vmap(cluster)(u, J, jax.random.split(kp, W))
                elif cont:
                    prop = u.at[ids, i].add(jax.random.uniform(kp, (W,), minval=-sw, maxval=sw))
                    prop = jnp.mod(prop, 2*jnp.pi)
                else:
                    prop = u.at[ids, i].set(1-u[ids, i])
                proposed_logs = batch_log(phi, coeff_batch(prop))
                delta = 2*(proposed_logs.real-logs.real)
                if method == 'conditional':
                    # Exact probability of the other value; retaining old value
                    # is an ordinary Gibbs draw, not a rejected Gibbs proposal.
                    probability = jax.nn.sigmoid(delta)
                    accepted = jax.random.uniform(ka, (W,)) < probability
                else:
                    accepted = jnp.log(jax.random.uniform(ka, (W,))) < cluster_accept_log(delta, proxy_delta)
                return (jnp.where(accepted[:, None], prop, u),
                        jnp.where(accepted, proposed_logs, logs), key,
                        changes+jnp.mean(accepted), sizes+jnp.mean(size))

            u, logs, key, changed, sizes = jax.lax.fori_loop(0, N, spin, (u, logs, key, 0., 0.))
            return (r, u, phi, key), jnp.array([jnp.mean(accept), changed/N, sizes/N])

        @jax.jit
        def chunk(p, state, rw, sw, coupling):
            def body(carry, _):
                return sweep(p, carry, rw, sw, coupling)
            state, stats = jax.lax.scan(body, state, None, length=sweeps)
            return state, jnp.mean(stats, axis=0)
        self.chunks[cache_key] = chunk
        return chunk
