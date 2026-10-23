"""Launch one convergence-validation replica per GPU; persist new outputs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from launch import sync_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=int, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--remote", type=Path, required=True)
    args = parser.parse_args()
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    if len(devices) != 8 or not 0 <= args.host < 6:
        raise ValueError("Requires six hosts with eight GPUs each.")
    available = sorted(os.sched_getaffinity(0))
    groups = [available[i::8] for i in range(8)]
    args.out.mkdir(parents=True, exist_ok=True)
    args.remote.mkdir(parents=True, exist_ok=True)
    active = []
    failure = None
    try:
        for index, device in enumerate(devices):
            rank = args.host*8+index
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device,
                       OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1")
            log = (args.out/f"rank{rank:02d}.log").open("w")
            process = subprocess.Popen([sys.executable, "-u",
                str(Path(__file__).with_name("revalidate.py")),
                "--rank", str(rank), "--checkpoint-root", str(args.checkpoint_root),
                "--out", str(args.out)], env=env, stdout=log,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda group=groups[index]: os.sched_setaffinity(0, group))
            active.append((rank, process, log))
        last_sync = 0.
        while any(p.poll() is None for _, p, _ in active):
            for rank, process, _ in active:
                if process.poll() not in (None, 0):
                    failure = dict(rank=rank, returncode=process.returncode)
                    raise RuntimeError(f"Validation replica failed: {failure}")
            if time.monotonic()-last_sync > 60:
                sync_files(args.out, args.remote)
                last_sync = time.monotonic()
            time.sleep(2)
        for rank, process, _ in active:
            if process.returncode != 0:
                failure = dict(rank=rank, returncode=process.returncode)
                raise RuntimeError(f"Validation replica failed: {failure}")
    finally:
        for _, process, log in active:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.close()
        (args.out/"host_status.json").write_text(json.dumps(dict(
            host=args.host, failure=failure, execution_finished=failure is None,
            scientific_convergence=False), indent=2))
        sync_files(args.out, args.remote)


if __name__ == "__main__":
    main()
