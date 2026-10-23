"""Resume immutable first-run checkpoints for mixing and plateau validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import arviz_stats as az
import jax
import jax.numpy as jnp
import numpy as np

from analyze import finite_json
from convergence_stats import diagnose, plateau
from physics import microbatch_map
from physics import Model
from run import SR, atomic_json, manifest


def q_grid(model):
    primitive = np.array([[1, 0], [0, 1], [-1, 1]])
    cdw = np.array([[1/3, 1/3], [2/3, -1/3], [-1/3, 2/3]])
    indices = np.rint(np.vstack([[0, 0], cdw, primitive]) @
                     np.asarray(model.cfg.supercell_matrix).T).astype(int)
    return indices, jnp.asarray(indices) @ (2*jnp.pi*model.invcell.T)


def frozen(model, params, training_state, sampler, width, step, args, out, energy_batch):
    """Fresh dispersed banks; fixed parameters throughout each long trace."""
    r, s, key = training_state
    key = jax.random.fold_in(key, step+20261023)
    key, a, b, c = jax.random.split(key, 4)
    half = model.walkers//2
    r = r.at[half:].set(jax.random.uniform(a, r[half:].shape) @ model.cell)
    s = jax.random.uniform(b, s.shape)*2*jnp.pi
    state = r, s, c
    started = time.perf_counter()
    phase = f"frozen_{step}"

    def progress(stage, count, **extra):
        atomic_json(out/"progress.json", finite_json(dict(
            phase=phase, stage=stage, step=step, records=count,
            seconds=time.perf_counter()-started, **extra)))

    for index in range(args.burnin):
        state, acceptance = sampler(params, state, width)
        if (index+1) % 100 == 0:
            jax.block_until_ready(state)
            progress("thermalizing", 0, burnin_done=index+1,
                     acceptance=np.asarray(acceptance).tolist())
    indices, q = q_grid(model)
    charge = jax.jit(lambda r:
        jnp.exp(1j*jnp.einsum("wnd,qd->wnq", r, q)).sum(axis=1))
    energies, densities = [], []
    primary = args.mid_records if step == 10000 else args.records
    maximum = args.max_records
    target = primary
    result = None
    while True:
        while len(energies) < target:
            state, acceptance = sampler(params, state, width)
            values = np.asarray(energy_batch(params, state[0], state[1]))
            density = np.asarray(charge(state[0]))
            if not np.isfinite(values).all() or not np.isfinite(density).all():
                raise FloatingPointError(f"Nonfinite {phase} observation.")
            energies.append(values)
            densities.append(density)
            count = len(energies)
            if count % 128 == 0:
                progress("measuring", count, target_records=target,
                         acceptance=np.asarray(acceptance).tolist())
            if count in [512, 1024, 2048, 4096, 8192]:
                x = np.asarray(energies)[:, :, 0].real
                prefix = dict(records=count,
                    energy_rank_rhat=float(az.rhat(x.T)),
                    energy_mean=float(x.mean()))
                atomic_json(out/f"{phase}_prefix_{count}.json", finite_json(prefix))
                print("PREFIX", step, json.dumps(finite_json(prefix)), flush=True)
        values, density = np.asarray(energies), np.asarray(densities)
        progress("diagnosing", len(values), target_records=target)
        result = diagnose(values, density, indices, model.n)
        result.update(step=step, records=len(values), walkers=model.walkers,
            burnin_calls=args.burnin, sweeps_per_call=args.sweeps,
            energy_microbatch=args.energy_microbatch,
            q_indices=indices.tolist(), energy_unclipped=True,
            storage="complex energy components, K/primitive rho, final sampling bank",
            seconds=time.perf_counter()-started)
        atomic_json(out/f"{phase}.json", finite_json(result))
        print("FROZEN", step, json.dumps(dict(
            records=result["records"], mixing_pass=result["mixing_pass"],
            mean=result["energy"]["statistics"]["mean"],
            se=result["energy"]["statistics"]["se"],
            energy_rank_rhat=result["energy"]["statistics"]["rank_rhat"],
            seconds=result["seconds"])), flush=True)
        if result["mixing_pass"] or target >= maximum:
            break
        target = min(2*target, maximum)
        print("EXTEND_FROZEN", step, target, flush=True)
    path = out/f"{phase}.npz"
    with path.with_suffix(".tmp").open("wb") as file:
        np.savez_compressed(file, energies=values, rho=density, q_indices=indices,
            cell=np.asarray(model.cell), final_positions=np.asarray(state[0]),
            final_angles=np.asarray(state[1]), final_key=np.asarray(state[2]))
    path.with_suffix(".tmp").replace(path)
    progress("saved", len(values), mixing_pass=result["mixing_pass"])
    return result


def validate_gpu(model, case, params, state, width, sampler, energy_batch):
    for kernel in ["full", "mixture"]:
        old = model.make_sampler(20, cache_orbitals=False, spatial_kernel=kernel)
        cached20 = model.make_sampler(20, spatial_kernel=kernel)
        first = jax.block_until_ready(old(params, state, width))
        second = jax.block_until_ready(cached20(params, state, width))
        for expected, actual in zip(jax.tree.leaves(first), jax.tree.leaves(second)):
            np.testing.assert_array_equal(expected, actual)
    scan = Model(theta=case["theta"], filling=case["filling"], cells=27,
                 width=32, depth=3, cutoff=9, walkers=model.walkers,
                 kinetic_mode="scan")
    e1 = np.asarray(jax.jit(model.energy)(params, state[0][0], state[1][0]))
    e2 = np.asarray(jax.jit(scan.energy)(params, state[0][0], state[1][0]))
    np.testing.assert_allclose(e1, e2, atol=1e-7, rtol=1e-8)
    reference = np.asarray(model.energy_batch(params, state[0], state[1]))
    measured = np.asarray(energy_batch(params, state[0], state[1]))
    np.testing.assert_allclose(measured, reference, atol=1e-7, rtol=1e-8)
    timings = []
    for _ in range(3):
        begin = time.perf_counter()
        bank, _ = sampler(params, state, width)
        jax.block_until_ready(energy_batch(params, bank[0], bank[1]))
        timings.append(time.perf_counter()-begin)
    return dict(exact_sampler_trajectory_parity=True,
        forward_scan_max_error=float(np.max(abs(e1-e2))),
        frozen_energy_batch_max_error=float(np.max(abs(measured-reference))),
        sample_and_energy_seconds=timings,
        steady_sample_and_energy_seconds=float(np.median(timings[1:])),
        real_gpu=all(x.platform == "gpu" for x in jax.devices()))


def execute(args):
    case = manifest()[args.rank]
    name = (f"rank{args.rank:02d}_theta{case['theta']}_nu{case['filling']}over3_"
            f"{case['motif']}_seed{case['seed']}")
    source = args.checkpoint_root/name
    checkpoint = source/"latest.pkl"
    payload = checkpoint.read_bytes()
    record = pickle.loads(payload)
    if (record["case"] != case or record["step"] != 8000 or
            record["architecture"] != [32, 3, 9] or record["state"][0].shape[0] != 128):
        raise ValueError("Expected matching, completed first-run checkpoint.")
    out = args.out/name
    out.mkdir(parents=True, exist_ok=True)
    model = Model(theta=case["theta"], filling=case["filling"], cells=27,
                  motif=case["motif"], width=32, depth=3, cutoff=9, walkers=128)
    params = jax.tree.map(jnp.asarray, record["params"])
    state = tuple(jnp.asarray(x) for x in record["state"])
    width = record["width"]
    sampler = model.make_sampler(args.sweeps, spatial_kernel=args.spatial_kernel)
    energy_batch = jax.jit(lambda p, r, s: microbatch_map(
        lambda x, y: model.energy(p, x, y), r, s, args.energy_microbatch))
    metadata = dict(case=case, input_checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
        checkpoint_step=record["step"], architecture=record["architecture"],
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        original_provenance=json.loads((source/"provenance.json").read_text()),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in Path(__file__).parent.glob("*.py")},
        devices=[str(x) for x in jax.devices()], jax=jax.__version__,
        physical_model="R1 unchanged; continuous angles; original SR",
        scientific_convergence=False)
    atomic_json(out/"provenance.json", metadata)
    gate = validate_gpu(model, case, params, state, width, sampler, energy_batch)
    atomic_json(out/"gpu_gate.json", gate)
    print("GPU_GATE", json.dumps(gate), flush=True)
    baseline = frozen(model, params, state, sampler, width, 8000, args, out, energy_batch)
    previous = json.loads((source/"evaluations.json").read_text())[-1]
    atomic_json(out/"original_comparison.json", dict(
        checkpoint_fixed=True,
        mean_difference=baseline["energy"]["statistics"]["mean"]-previous["mean"],
        original_se=previous["se"], new_se=baseline["energy"]["statistics"]["se"],
        original_mixing_pass=previous["mixing_certified"],
        new_mixing_pass=baseline["mixing_pass"]))
    sr = SR(model, params)
    history = []
    phases = {8000: baseline}
    begin = time.perf_counter()

    def save(step):
        with (out/"latest.tmp").open("wb") as file:
            pickle.dump(dict(case=case, architecture=[32, 3, 9], step=step,
                params=jax.device_get(params), state=jax.device_get(state),
                width=width, history=history), file, protocol=5)
        (out/"latest.tmp").replace(out/"latest.pkl")

    save(8000)
    for step in range(8000, args.steps):
        state, acceptance = sampler(params, state, width)
        params, metrics = sr.step(params, state[0], state[1], .02/(1+step/2000))
        if (step+1) % 20 == 0:
            values = np.asarray(metrics)
            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite SR metrics.")
            entry = dict(step=step+1, mean=float(values[0]), std=float(values[1]),
                imaginary_mean=float(values[2]), seconds=time.perf_counter()-begin,
                acceptance=np.asarray(acceptance).tolist(), width=width)
            history.append(entry)
            atomic_json(out/"history.json", history)
            atomic_json(out/"progress.json", dict(phase="training", **entry))
            print("TRAIN", json.dumps(entry), flush=True)
        if (step+1) % 200 == 0:
            save(step+1)
        if step+1 in [10000, 12000]:
            phases[step+1] = frozen(model, params, state, sampler, width,
                                    step+1, args, out, energy_batch)
            save(step+1)
    check = plateau(phases[10000], phases[12000])
    atomic_json(out/"plateau.json", check)
    atomic_json(out/"status.json", dict(status="validation_budget_finished",
        steps=args.steps, final_mixing_pass=phases[12000]["mixing_pass"],
        fixed_capacity_plateau_pass=check["fixed_capacity_plateau_pass"],
        scientific_convergence=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--spatial-kernel", choices=["full", "mixture"], default="mixture")
    parser.add_argument("--burnin", type=int, default=1000)
    parser.add_argument("--records", type=int, default=4096)
    parser.add_argument("--mid-records", type=int, default=4096)
    parser.add_argument("--max-records", type=int, default=8192)
    parser.add_argument("--energy-microbatch", type=int, default=32,
                        choices=[4, 8, 16, 32, 64, 128])
    args = parser.parse_args()
    if (not 0 <= args.rank < 48 or args.steps != 12000 or args.burnin < 1000 or
            min(args.records, args.mid_records) < 4096 or
            args.max_records < max(args.records, args.mid_records)):
        raise ValueError("Use the declared 48-rank, 12000-step validation budget.")
    if any(device.platform != "gpu" for device in jax.devices()):
        raise RuntimeError("Validation requires a real GPU.")
    execute(args)


if __name__ == "__main__":
    main()
