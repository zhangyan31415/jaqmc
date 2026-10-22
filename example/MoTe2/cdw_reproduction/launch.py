"""One independent replica per assigned GPU, with periodic HDFS snapshots."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def sync_files(source, target):
    for file in source.rglob("*"):
        if not file.is_file() or file.suffix == ".tmp":
            continue
        dest = target/file.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            if dest.exists() and dest.stat().st_size == file.stat().st_size and dest.stat().st_mtime >= file.stat().st_mtime:
                continue
            shutil.copyfile(file, dest)
        except (OSError, FileNotFoundError) as error:
            print("SYNC_RETRY", file, str(error), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--remote", type=Path, required=True)
    parser.add_argument("--host", type=int, required=True)
    parser.add_argument("--steps", type=int, default=8000)
    args = parser.parse_args()
    ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    if len(ids) != 8 or not 0 <= args.host < 6:
        raise ValueError("Production requires six hosts with eight GPUs each.")
    args.out.mkdir(parents=True, exist_ok=True)
    args.remote.mkdir(parents=True, exist_ok=True)
    active = []
    available = sorted(os.sched_getaffinity(0))
    cpu_groups = [available[i::8] for i in range(8)]
    for index, gpu in enumerate(ids):
        rank = 8*args.host+index
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, NQS_RANK=str(rank),
                   OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1")
        log = (args.out/f"rank{rank:02d}.log").open("w")
        process = subprocess.Popen([sys.executable, "-u",
            str(Path(__file__).with_name("run.py")), "train", "--out", str(args.out),
            "--rank", str(rank), "--steps", str(args.steps)], env=env,
            stdout=log, stderr=subprocess.STDOUT,
            preexec_fn=lambda group=cpu_groups[index]: os.sched_setaffinity(0, group))
        active.append((rank, process, log))
    failure = None
    last_sync = 0.
    try:
        while any(p.poll() is None for _, p, _ in active):
            for rank, process, _ in active:
                if process.poll() not in (None, 0):
                    failure = dict(rank=rank, returncode=process.returncode)
                    raise RuntimeError(f"Replica failed: {failure}")
            if time.monotonic()-last_sync > 60:
                sync_files(args.out, args.remote)
                last_sync = time.monotonic()
            time.sleep(2)
        for rank, process, _ in active:
            if process.returncode != 0:
                failure = dict(rank=rank, returncode=process.returncode)
                raise RuntimeError(f"Replica failed: {failure}")
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
