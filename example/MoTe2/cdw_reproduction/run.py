"""Validation, independent SR training, frozen measurements, and manifest.

All reported energies are unclipped, in meV/hole. Energy clipping is used
only to stabilize optimization. A finished allocation is not convergence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import socket
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from physics import Model, microbatch_map
from reference import PAPER, configuration, hole_model


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def manifest():
    return [dict(rank=i, theta=theta, filling=filling, motif=motif, seed=seed,
                 cells=27, physics="R1_full_continuum")
            for i, (theta, filling, motif, seed) in enumerate(
                (t, n, m, s) for t in [3., 4.2] for n in [1, 2]
                for m in ["MX", "XM", "MM"] for s in [731, 834, 937, 1040])]


class SR:
    """Real-parameter amplitude+phase SR, with a sample-space linear solve."""
    def __init__(self, model, params, damping=.05):
        self.model, self.damping = model, damping
        _, self.unravel = ravel_pytree(params)
        self.step = jax.jit(self.update)

    def update(self, p, r, s, lr):
        m = self.model
        energy = m.energy_batch(p, r, s)[:, 0]
        median = jnp.median(energy.real)
        scale = jnp.maximum(jnp.mean(abs(energy.real-median)), .1)
        imag_med = jnp.median(energy.imag)
        imag_scale = jnp.maximum(jnp.mean(abs(energy.imag-imag_med)), .1)
        clipped = jnp.clip(energy.real, median-5*scale, median+5*scale) + 1j*jnp.clip(
            energy.imag, imag_med-5*imag_scale, imag_med+5*imag_scale)
        x, _ = ravel_pytree(p)
        def derivative(a, b):
            def logs(v):
                z = m.log(self.unravel(v), a, b)
                return jnp.array([z.real, z.imag])
            return jax.jacrev(logs)(x)
        jac = microbatch_map(derivative, r, s, 4)
        score = jac[:, 0, :] + 1j*jac[:, 1, :]
        centered = score-score.mean(axis=0)
        e = clipped-clipped.mean()
        a = jnp.concatenate([centered.real, centered.imag])/jnp.sqrt(len(e))
        b = jnp.concatenate([e.real, e.imag])/jnp.sqrt(len(e))
        dual = a@a.T + self.damping*jnp.eye(len(b))
        direction = a.T@jnp.linalg.solve(dual, b)
        delta = lr*direction
        metric = jnp.sqrt(jnp.mean(abs(centered@delta)**2))
        delta *= jnp.minimum(1., .05/jnp.maximum(metric, 1e-12))
        delta *= jnp.minimum(1., .15/jnp.maximum(jnp.linalg.norm(delta), 1e-12))
        return self.unravel(x-delta), jnp.array([
            energy.real.mean(), energy.real.std(), energy.imag.mean(),
            jnp.linalg.norm(delta), metric])


def provenance(args):
    root = Path(__file__).resolve().parents[3]
    source = {}
    for path in sorted(list((root/"src").rglob("*.py")) +
                       list(Path(__file__).parent.glob("*.py"))):
        source[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(host=socket.gethostname(), python=sys.version, jax=jax.__version__,
        devices=[str(x) for x in jax.devices()], parameters=vars(args) |
        {"out": str(args.out)}, source_sha256=source, paper_parameters=PAPER,
        physics="R1 full continuum, not R0 adiabatic LL Hamiltonian",
        energy_zero="Eref=0; no comparison of absolute energies to paper HF",
        command=sys.argv)


def frozen_measurement(m, p, state, sampler, width, records, out):
    """Two independently thermalized banks; stored complex energy and rho(q).

    The simulation-reciprocal grid includes CDW K vectors and primitive
    harmonics. Individual walker trajectories remain intact in the archive.
    """
    r, s, key = state
    key, a, b, c = jax.random.split(key, 4)
    # Half start near the training bank and half from uniform positions.
    half = m.walkers//2
    r = r.at[half:].set(jax.random.uniform(a, r[half:].shape)@m.cell)
    s = jax.random.uniform(b, s.shape)*2*jnp.pi
    state = r, s, c
    for _ in range(100):
        state, _ = sampler(p, state, width)
    indices = np.array([(u, v) for u in range(-6, 7) for v in range(-6, 7)])
    q = jnp.asarray(indices) @ (2*jnp.pi*m.invcell.T)
    def observables(r):
        return jnp.exp(1j*jnp.einsum("wnd,qd->wnq", r, q)).sum(axis=1)
    charge_fn = jax.jit(observables)
    values, densities, positions, angles = [], [], [], []
    for _ in range(records):
        state, _ = sampler(p, state, width)
        e = np.asarray(m.energy_batch(p, state[0], state[1]))
        if not np.isfinite(e).all():
            raise FloatingPointError("Nonfinite frozen local energies.")
        values.append(e)
        densities.append(np.asarray(charge_fn(state[0])))
        positions.append(np.asarray(state[0]))
        angles.append(np.asarray(state[1]))
    values, densities = np.asarray(values), np.asarray(densities)
    np.savez_compressed(out, energies=values, rho=densities, q_indices=indices,
                        cell=np.asarray(m.cell), positions=np.asarray(positions),
                        angles=np.asarray(angles))
    e = values[:, :, 0].real
    block = min(32, records//4)
    block_count = records//block
    block_means = e[:block_count*block].reshape(block_count, block, m.walkers).mean(axis=1)
    se = block_means.std(ddof=1)/np.sqrt(block_count*m.walkers)
    family = [float(e[:, :half].mean()), float(e[:, half:].mean())]
    walker_se = e.mean(axis=0).std(ddof=1)/np.sqrt(m.walkers)
    se = max(float(se), float(walker_se))
    drift = float(e[records//2:].mean()-e[:records//2].mean())
    summary = dict(mean=float(e.mean()), se=se, variance=float(e.var()),
        imaginary_mean=float(values[:, :, 0].imag.mean()), records=records,
        family_means=family, half_drift=drift, trace=str(out),
        # Necessary stationarity checks, not sufficient certification of mixing.
        stationarity_pass=bool(abs(family[0]-family[1]) < 6*se+.01 and
                               abs(drift) < 6*se+.01),
        mixing_certified=False, energy_unclipped=True)
    if records >= 128:
        from mcstats import split_rhat, summary as trace_summary
        stats = trace_summary(e)
        summary["statistics"] = stats
        summary["se"] = max(se, stats["ac"]["se"],
                            *stats["block_se"].values())
        k_indices = np.rint(np.array([[1/3, 1/3], [2/3, -1/3], [-1/3, 2/3]]) @
                            np.asarray(m.cfg.supercell_matrix).T).astype(int)
        charge_rhat = []
        for index in k_indices:
            column = np.flatnonzero(np.all(indices == index, axis=1))[0]
            trace = densities[:, :, column]/m.n
            charge_rhat.extend(split_rhat(v)["rank"] for v in [
                trace.real, trace.imag, abs(trace)])
        summary["charge_rank_rhat"] = charge_rhat
        summary["mixing_certified"] = bool(
            summary["stationarity_pass"] and stats["rhat"]["rank"] < 1.01 and
            max(charge_rhat) < 1.02)
    atomic_json(out.with_suffix(".json"), summary)
    return summary


def validate(args):
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out/"provenance.json", provenance(args))
    gates = []
    begin = time.perf_counter()
    # Exact free Slater control tests the sum of kinetic, SOC, and tunneling.
    for filling in [1, 2]:
        m = Model(cells=3, filling=filling, width=4, depth=1, walkers=4,
                  cutoff=5, coulomb=False, free_reference=True)
        p, r, s = m.initialize(4242+filling)
        values = np.asarray(m.energy_batch(p, r, s))[:, 0]
        reference = m.exact_energy/m.n
        error = float(np.max(abs(values-reference)))
        gate = dict(kind="U0_exact_Slater", filling=filling, cells=3,
                    reference_mev_per_hole=reference, max_local_error=error,
                    tolerance=.02, passed=error < .02)
        gates.append(gate)
        print("VALIDATE", json.dumps(gate), flush=True)
    # Real pilot geometry with the production neural architecture.
    for filling in [1, 2]:
        m = Model(cells=12, filling=filling, width=args.width, depth=args.depth,
                  walkers=args.walkers, cutoff=args.cutoff)
        p, r, s = m.initialize(5100+filling)
        scan = Model(cells=12, filling=filling, width=args.width, depth=args.depth,
                     walkers=args.walkers, cutoff=args.cutoff, kinetic_mode="scan")
        e1 = np.asarray(jax.jit(m.energy)(p, r[0], s[0]))
        e2 = np.asarray(jax.jit(scan.energy)(p, r[0], s[0]))
        error = float(np.max(abs(e1-e2)))
        sr = SR(m, p)
        sampler = m.make_sampler(args.sweeps)
        state = r, s, jax.random.PRNGKey(5100+filling+10000)
        metrics = []
        for step in range(args.steps):
            state, acceptance = sampler(p, state, .1)
            p, metric = sr.step(p, state[0], state[1], .01)
            metrics.append(np.asarray(metric).tolist())
        gate = dict(kind="interacting_SR_pilot", filling=filling, cells=12,
            holes=m.n, kinetic_max_error=error, sr_steps=args.steps,
            acceptance=np.asarray(acceptance).tolist(), metrics=metrics,
            passed=bool(error < 1e-7 and np.isfinite(metrics).all()))
        gates.append(gate)
        print("VALIDATE", json.dumps(gate), flush=True)
        hole_model(m.cfg).save(args.out/f"model_theta3_nu{filling}over3.npz")
    passed = all(x["passed"] for x in gates)
    atomic_json(args.out/"validation.json", dict(passed=passed, gates=gates,
        seconds=time.perf_counter()-begin, requires_gpu=not args.allow_cpu,
        real_gpu=all(x.platform=="gpu" for x in jax.devices())))
    if not passed:
        raise RuntimeError("Scientific validation gate failed; production prohibited.")


def train(args):
    rank = args.rank if args.rank is not None else int(os.environ.get("NQS_RANK", "0"))
    case = manifest()[rank]
    out = args.out/f"rank{rank:02d}_theta{case['theta']}_nu{case['filling']}over3_{case['motif']}_seed{case['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/"provenance.json", provenance(args) | {"case": case})
    m = Model(theta=case["theta"], filling=case["filling"], cells=27,
              motif=case["motif"], walkers=args.walkers, width=args.width,
              depth=args.depth, cutoff=args.cutoff)
    hole_model(m.cfg).save(out/"model.npz")
    p, r, s = m.initialize(case["seed"])
    state = r, s, jax.random.PRNGKey(case["seed"]+10000)
    sampler = m.make_sampler(args.sweeps)
    width = .15/np.sqrt(m.n)
    sr = SR(m, p)
    history, evaluations = [], []
    begin = time.perf_counter()
    start = 0
    latest = out/"latest.pkl"
    if latest.exists():
        with latest.open("rb") as f:
            previous = pickle.load(f)
        if previous["case"] != case or previous["architecture"] != [args.width, args.depth, args.cutoff]:
            raise ValueError("Resume configuration mismatch.")
        p = jax.tree.map(jnp.asarray, previous["params"])
        state = tuple(jnp.asarray(x) for x in previous["state"])
        start, width = previous["step"], previous["width"]
        history, evaluations = previous["history"], previous["evaluations"]
    else:
        for step in range(100):
            state, acceptance = sampler(p, state, width)
            width = float(np.clip(width*np.exp(.05*(float(acceptance[0])-.45)), .005, .25))
    def save(step):
        record = dict(case=case, architecture=[args.width, args.depth, args.cutoff],
            params=jax.device_get(p), state=jax.device_get(state), step=step,
            width=width, history=history, evaluations=evaluations)
        tmp = latest.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(record, f, protocol=5)
        tmp.replace(latest)
    for step in range(start, args.steps):
        state, acceptance = sampler(p, state, width)
        lr = .02/(1+step/2000)
        p, metric = sr.step(p, state[0], state[1], lr)
        values = np.asarray(metric)
        if not np.isfinite(values).all():
            raise FloatingPointError(f"Nonfinite SR at rank {rank}, step {step}.")
        if (step+1) % 20 == 0:
            entry = dict(step=step+1, mean=float(values[0]), std=float(values[1]),
                imag=float(values[2]), delta_norm=float(values[3]), metric=float(values[4]),
                acceptance=np.asarray(acceptance).tolist(), width=width, lr=lr,
                seconds=time.perf_counter()-begin)
            history.append(entry)
            print("TRAIN", json.dumps(entry), flush=True)
            atomic_json(out/"history.json", history)
        if (step+1) % 200 == 0:
            save(step+1)
        if (step+1) % args.eval_every == 0:
            ev = frozen_measurement(m, p, state, sampler, width, args.records,
                                    out/f"frozen_{step+1}.npz")
            ev["step"] = step+1
            evaluations.append(ev)
            print("FROZEN", json.dumps(ev), flush=True)
            atomic_json(out/"evaluations.json", evaluations)
            save(step+1)
    save(args.steps)
    atomic_json(out/"status.json", dict(status="training_budget_finished",
        steps=args.steps, converged=False,
        reason="Requires frozen-trace autocorrelation/mixing and cross-seed plateau review."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["manifest", "validate", "train"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--cutoff", type=int, default=9)
    parser.add_argument("--walkers", type=int, default=128)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--records", type=int, default=512)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.mode == "manifest":
        atomic_json(args.out, manifest())
        return
    if not args.allow_cpu and any(x.platform != "gpu" for x in jax.devices()):
        raise RuntimeError("GPU validation/production requires a real GPU device.")
    if args.walkers % 4 or args.records < 8:
        raise ValueError("Use walker counts divisible by 4 and >=8 frozen records.")
    (validate if args.mode == "validate" else train)(args)


if __name__ == "__main__":
    main()
