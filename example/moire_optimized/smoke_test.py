# Copyright (c) 2026 JaQMC contributors
# SPDX-License-Identifier: Apache-2.0

"""Reduced CPU workflow/continuation checks; not a material calculation."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from jaqmc.app.moire.optimized.runner import HERE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONHASHSEED="731")
    for name in ("mote2", "gamma"):
        cfg = json.loads((HERE / "configs" / f"{name}.json").read_text())
        cfg["system"].update(
            nelec=2,
            electron_spins=[2, 0] if name == "mote2" else [1, 1],
            supercell_matrix=[[1, 0], [0, 1]],
        )
        cfg["wf"].update(hidden_dims_single=[8, 8], hidden_dims_double=[4, 4])
        (out / f"{name}_small.json").write_text(json.dumps(cfg, indent=2) + "\n")
    commands = []
    cases = [
        ("mote_full3", "mote2", 3, []),
        ("mote_split1", "mote2", 1, []),
        (
            "mote_resume3",
            "mote2",
            2,
            ["--resume", str(out / "mote_split1/train_ckpt_000000.npz")],
        ),
        ("mote_discrete2", "mote2", 2, ["--representation", "discrete"]),
        ("gamma2", "gamma", 2, []),
    ]
    for case, model, steps, extra in cases:
        command = [
            sys.executable,
            "-m",
            "jaqmc.app.moire.optimized.runner",
            "--model",
            model,
            "--config",
            str(out / f"{model}_small.json"),
            "--out",
            str(out / case),
            "--steps",
            str(steps),
            "--batch",
            "4",
            "--burn",
            "2",
            "--checkpoint-every",
            "1",
            "--cache-dir",
            str(out / ("cache_" + model)),
            *extra,
        ]
        commands.append(command)
        print("RUN", case, flush=True)
        with (out / f"{case}.log").open("w") as log:
            subprocess.run(
                command,
                cwd=out,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    checked = {}
    for case, _, _, _ in cases:
        meta = json.loads((out / case / "launch.json").read_text())
        assert meta["state"] == "completed", meta
        checked[case] = {
            k: meta[k]
            for k in ("completed_updates", "optimizer_counts", "burn_performed")
        }
    with (
        np.load(out / "mote_full3/train_ckpt_000002.npz") as a,
        np.load(out / "mote_resume3/train_ckpt_000002.npz") as b,
    ):
        assert set(a.files) == set(b.files)
        different = [k for k in a.files if not np.array_equal(a[k], b[k])]
        assert not different, different
        arrays = len(a.files)
    assert not checked["mote_resume3"]["burn_performed"]
    with np.load(out / "mote_discrete2/train_ckpt_000001.npz") as data:
        assert np.isin(data["batched_data/data/spin_coords"], [0, 1]).all()
    result = dict(
        scope=(
            "Reduced N2/batch4 CPU smoke only; "
            "no production performance or convergence claim"
        ),
        cases=checked,
        continuation_all_arrays_bitwise_equal=True,
        compared_arrays=arrays,
        commands=commands,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(state="passed", cases=len(cases), compared_arrays=arrays, out=str(out))
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
