# Copyright (c) 2026 JaQMC contributors
# SPDX-License-Identifier: Apache-2.0

"""Portable single-device training with full native state and PRNG continuation."""

import argparse
import csv
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def create_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["mote2", "gamma"], default="mote2")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--gamma-model", type=Path, help="Portable six-parameter model JSON")
    p.add_argument(
        "--steps",
        type=int,
        default=200,
        help="ADDITIONAL optimizer updates, also on resume",
    )
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--seed", type=int, default=731)
    p.add_argument(
        "--burn",
        type=int,
        default=1000,
        help="Cold-start sampler burn; skipped on resume",
    )
    p.add_argument(
        "--representation", choices=["continuous", "discrete"], default="continuous"
    )
    p.add_argument("--derivative", choices=["baseline", "jvp", "lap", "sparse-lap"])
    p.add_argument("--directions", type=int, default=6)
    p.add_argument("--resume", type=Path)
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config; construct no network",
    )
    return p


def gamma_model_path(args):
    return (args.gamma_model or HERE / "models/zrs2_gamma_3p15.json").resolve()


def build_config(args):
    if (
        min(args.steps, args.batch, args.chunk, args.directions, args.checkpoint_every)
        < 1
        or args.burn < 0
    ):
        raise ValueError(
            "steps, batch, chunk, directions and checkpoint interval must be positive; "
            "burn nonnegative"
        )
    derivative = args.derivative or ("sparse-lap" if args.model == "mote2" else "jvp")
    if args.model == "gamma" and args.representation != "continuous":
        raise ValueError("Gamma does not support this binary discrete representation")
    if args.model == "mote2" and derivative == "jvp":
        raise ValueError("MoTe2 derivative must be baseline, lap or sparse-lap")
    path = args.config or HERE / "configs" / (args.model + ".json")
    cfg = json.loads(path.read_text())
    cfg["workflow"].pop("restore_path", None)
    cfg["workflow"].update(
        save_path=str(args.out.resolve()), batch_size=args.batch, seed=args.seed
    )
    cfg["train"]["run"].update(iterations=50000, burn_in=args.burn)
    cfg["train"]["optim"]["inverse_update_period"] = 1
    if args.model == "gamma":
        from .gamma_model import load_model

        length_nm = load_model(gamma_model_path(args))["length_A"] / 10
        if (
            args.config
            and abs(cfg["system"]["moire_lattice_constant_nm"] - length_nm) > 1e-7
        ):
            raise ValueError("Gamma config length does not match the supplied model")
        cfg["system"]["moire_lattice_constant_nm"] = length_nm
    return cfg


def config_fingerprint(cfg):
    cfg = json.loads(json.dumps(cfg))
    cfg["workflow"].pop("save_path", None)
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _contract(args, cfg):
    return dict(
        model=args.model,
        representation=args.representation,
        derivative=args.derivative,
        directions=args.directions,
        chunk=args.chunk,
        config=config_fingerprint(cfg),
        gamma_model_sha256=hashlib.sha256(
            gamma_model_path(args).read_bytes()
        ).hexdigest()
        if args.model == "gamma"
        else None,
    )


def _workflow(args, cfg):
    from jaqmc.app.moire.workflow import MoireTrainWorkflow
    from jaqmc.utils.config import ConfigManager
    from jaqmc.utils.runtime import configure_runtime

    from . import mote2
    from .checks import RawFiniteEstimator

    manager = ConfigManager(cfg)
    configure_runtime(manager)
    mote2.prepare_representation(args.representation)
    wf = MoireTrainWorkflow(manager)
    stage = wf.train_stage
    if args.model == "mote2":
        mote2.install_backend(
            stage,
            wf.wf,
            args.chunk,
            args.representation,
            sparse=args.derivative == "sparse-lap",
        )
        if args.representation == "discrete":
            mote2.install_sampler(stage, wf.wf, position_steps=cfg["sampler"]["steps"])
    else:
        from .gamma import install_backend
        from .gamma_model import load_model

        model = load_model(gamma_model_path(args))
        install_backend(
            stage,
            wf.wf,
            model,
            model["length_A"],
            chunk=args.chunk,
            derivative_method=args.derivative,
            directions=args.directions,
        )
    for name, est in list(stage.estimators.estimators.items()):
        stage.estimators.estimators[name] = RawFiniteEstimator(est, name)
    wf.prepare()
    return wf


def _burn(stage, state, key, rounds):
    from dataclasses import replace

    import jax
    import jax.numpy as jnp

    from jaqmc.utils import parallel_jax as pj

    def run(state, key):
        def body(_, carry):
            state, key, bad, residual = carry
            key, sub = jax.random.split(key)
            data, stats, sampler = stage.sample_plan.step(
                state.params, state.batched_data, state.sampler_state, sub
            )
            bad += sum(
                jnp.sum(~jnp.isfinite(v)) for v in jax.tree.leaves((data, stats))
            )
            bad += sum(jnp.sum(v) for k, v in stats.items() if "nonfinite_count" in k)
            r = max(
                (
                    v
                    for k, v in stats.items()
                    if k.endswith("spin_inverse_residual_max")
                ),
                default=jnp.array(0.0),
            )
            return (
                replace(state, batched_data=data, sampler_state=sampler),
                key,
                bad,
                jnp.maximum(residual, r),
            )

        return jax.lax.fori_loop(
            0, rounds, body, (state, key, jnp.array(0), jnp.array(0.0))
        )

    p = pj.SHARE_PARTITION
    burn = pj.jit_sharded(
        run,
        in_specs=(state.partition(), p),
        out_specs=(state.partition(), p, p, p),
        check_vma=False,
    )
    state, key, bad, residual = burn(state, key)
    jax.block_until_ready((state, key, bad, residual))
    if int(bad) or float(residual) >= 1e-3:
        raise FloatingPointError(
            f"Invalid burn: count={bad}, inverse residual={residual}"
        )
    return state, key


def _run(args, cfg, meta):  # noqa: C901
    import jax
    import numpy as np

    from jaqmc.utils import parallel_jax as pj
    from jaqmc.workflow.base import init_batched_data

    from .checkpoint import load_checkpoint, optimizer_counts, save_checkpoint
    from .checks import check_stats

    if jax.device_count() != 1 or jax.process_count() != 1:
        raise ValueError("This example requires one visible device and process")
    started = time.perf_counter()
    wf = _workflow(args, cfg)
    stage = wf.train_stage
    key = jax.random.PRNGKey(args.seed)
    key, dk, pk, sk, lk = jax.random.split(key, 5)
    data = init_batched_data(wf.data_init, args.batch, dk)
    params = wf.wf.init_params(data.unbatched_example(), pk)
    if args.representation == "discrete":
        from .mote2 import initialize_labels

        data = initialize_labels(data, lk)
    state = stage.create_state(sk, params=params, batched_data=data)
    state = jax.device_put(state, pj.make_sharding(state.partition()))
    key = jax.device_put(key, pj.make_sharding(pj.SHARE_PARTITION))
    updates = 0
    if args.resume:
        sidecar = json.loads(args.resume.with_suffix(".json").read_text())
        if sidecar["contract"] != meta["contract"]:
            raise ValueError("Resume physical configuration or backend differs")
        if not sidecar["burn_completed"]:
            raise ValueError("Checkpoint burn is incomplete")
        if hashlib.sha256(args.resume.read_bytes()).hexdigest() != sidecar["sha256"]:
            raise ValueError("Resume checkpoint SHA256 differs")
        state, key, updates = load_checkpoint(args.resume, state)
    meta.update(
        start_update=updates,
        completed_updates=updates,
        burn_performed=not bool(args.resume),
        device=str(jax.devices()[0]),
        device_kind=jax.devices()[0].device_kind,
        jax=jax.__version__,
    )
    t = time.perf_counter()
    if not args.resume and args.burn:
        state, key = _burn(stage, state, key, args.burn)
    meta["burn_compile_execute_seconds"] = time.perf_counter() - t
    p = pj.SHARE_PARTITION
    step = pj.jit_sharded(
        stage.compute_step,
        in_specs=(state.partition(), p),
        out_specs=(state.partition(), p),
        check_vma=False,
    )
    t = time.perf_counter()
    unused, stats = step(state, jax.random.fold_in(key, 921))
    jax.block_until_ready((unused, stats))
    check_stats(stats)
    optimizer_counts(unused, updates + 1)
    meta["compile_disposable_warmup_seconds"] = time.perf_counter() - t
    del unused, stats

    def checkpoint():
        path = args.out / f"train_ckpt_{updates - 1:06d}.npz"
        save_checkpoint(path, state.process_allgather(), key, updates)
        sidecar = dict(
            contract=meta["contract"],
            burn_completed=True,
            completed_updates=updates,
            optimizer_counts=optimizer_counts(state, updates),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        _write_json(path.with_suffix(".json"), sidecar)
        meta["final_checkpoint"] = str(path)

    checkpoint()
    target = updates + args.steps
    total_step = 0.0
    fields = [
        "update",
        "energy_mev_per_hole",
        "imaginary_energy_mev_per_hole",
        "pmove",
        "raw_nonfinite_count",
        "sampler_inverse_residual_max",
        "step_seconds",
        "elapsed_seconds",
    ]
    with (args.out / "training.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        while updates < target:
            key, sub = jax.random.split(key)
            t = time.perf_counter()
            state, stats = step(state, sub)
            jax.block_until_ready((state, stats))
            elapsed = time.perf_counter() - t
            bad, residual = check_stats(stats)
            energy = complex(np.asarray(stats["total_energy"])) / cfg["system"]["nelec"]
            updates += 1
            total_step += elapsed
            writer.writerow(
                dict(
                    update=updates,
                    energy_mev_per_hole=energy.real,
                    imaginary_energy_mev_per_hole=energy.imag,
                    pmove=float(stats["pmove"]),
                    raw_nonfinite_count=bad,
                    sampler_inverse_residual_max=residual,
                    step_seconds=elapsed,
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
            if updates % args.checkpoint_every == 0 or updates == target:
                checkpoint()
            if updates % 100 == 0 or updates == target:
                stream.flush()
                meta.update(
                    completed_updates=updates,
                    synchronized_step_seconds=total_step,
                    elapsed_seconds=time.perf_counter() - started,
                )
                _write_json(args.out / "launch.json", meta)
                print("PROGRESS", updates, flush=True)
    meta.update(
        state="completed",
        completed_updates=updates,
        optimizer_counts=optimizer_counts(state, updates),
        synchronized_step_seconds=total_step,
        elapsed_seconds=time.perf_counter() - started,
    )


def main(argv=None):
    args = create_parser().parse_args(argv)
    args.derivative = args.derivative or (
        "sparse-lap" if args.model == "mote2" else "jvp"
    )
    cfg = build_config(args)
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    cache = (args.cache_dir or args.out / "compilation_cache").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    _write_json(args.out / "config.json", cfg)
    meta = dict(
        state="initializing",
        contract=_contract(args, cfg),
        host=socket.gethostname(),
        cwd=str(Path.cwd()),
        environment=sys.prefix,
        python=sys.executable,
        command=[sys.executable, *sys.argv],
        job_id=os.environ.get("SLURM_JOB_ID"),
        partition=os.environ.get("SLURM_JOB_PARTITION"),
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        threads={
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in HERE.glob("*.py")
        },
        timing=(
            "Synchronized steps are separate from whole process costs: "
            "guards, compilation, burn and checkpoint I/O"
        ),
    )
    _write_json(args.out / "launch.json", meta)
    try:
        _run(args, cfg, meta)
    except BaseException as error:
        meta.update(state="failed", error=repr(error))
        _write_json(args.out / "launch.json", meta)
        raise
    _write_json(args.out / "launch.json", meta)
    print(
        json.dumps(
            dict(
                state=meta["state"],
                completed_updates=meta["completed_updates"],
                out=str(args.out),
            )
        )
    )


if __name__ == "__main__":
    main()
