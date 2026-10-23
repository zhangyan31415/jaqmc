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
    parser.add_argument("--ranks", help="Optional subset, queued across available GPUs.")
    parser.add_argument("--gpu-count", type=int, help="Allocated GPUs when no device mask is set.")
    args = parser.parse_args()
    fallback = ",".join(map(str, range(args.gpu_count or 8)))
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", fallback).split(",")
    ranks = (list(map(int, args.ranks.split(","))) if args.ranks else
             list(range(args.host*8, args.host*8+8)))
    if (not devices or not 0 <= args.host < 6 or
            (args.gpu_count is not None and len(devices) != args.gpu_count) or
            (args.ranks is None and len(devices) != 8) or
            len(set(ranks)) != len(ranks) or any(not 0 <= r < 48 for r in ranks)):
        raise ValueError("Use valid unique ranks and one process per available GPU.")
    available = sorted(os.sched_getaffinity(0))
    groups = [available[i::len(devices)] for i in range(len(devices))]
    args.out.mkdir(parents=True, exist_ok=True)
    args.remote.mkdir(parents=True, exist_ok=True)
    active = {}
    pending = list(ranks)
    failure = None
    try:
        def start(index, rank):
            device = devices[index]
            checkpoint_root = (args.checkpoint_root/f"host{rank//8}" if args.ranks
                               else args.checkpoint_root)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device,
                       OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1")
            log = (args.out/f"rank{rank:02d}.log").open("w")
            process = subprocess.Popen([sys.executable, "-u",
                str(Path(__file__).with_name("revalidate.py")),
                "--rank", str(rank), "--checkpoint-root", str(checkpoint_root),
                "--out", str(args.out)], env=env, stdout=log,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda group=groups[index]: os.sched_setaffinity(0, group))
            active[index] = rank, process, log
        last_sync = 0.
        while pending or active:
            for index in range(len(devices)):
                if index not in active and pending:
                    start(index, pending.pop(0))
            for index, (rank, process, log) in list(active.items()):
                code = process.poll()
                if code not in (None, 0):
                    failure = dict(rank=rank, returncode=process.returncode)
                    raise RuntimeError(f"Validation replica failed: {failure}")
                if code == 0:
                    log.close()
                    del active[index]
            if time.monotonic()-last_sync > 60:
                sync_files(args.out, args.remote)
                last_sync = time.monotonic()
            time.sleep(2)
    except BaseException as error:
        failure = failure or dict(error=str(error))
        raise
    finally:
        for _, process, log in active.values():
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
            requested_ranks=ranks,
            scientific_convergence=False), indent=2))
        sync_files(args.out, args.remote)


if __name__ == "__main__":
    main()
