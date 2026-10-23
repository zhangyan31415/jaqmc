"""Create reviewable Merlin YAML and deterministic source archives.

Production YAML generation requires an observed, passing GPU validation JSON.
The command only prepares files; submission is an explicit separate command.
"""
from __future__ import annotations

import argparse
import base64
import copy
import gzip
import hashlib
import io
import json
import shlex
import tarfile
from pathlib import Path

import yaml


def archive(root):
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as bundle:
            paths = list((root/"src/jaqmc").rglob("*.py"))
            paths += list((root/"example/MoTe2/cdw_reproduction").rglob("*"))
            paths += [root/"pyproject.toml", root/"LICENSE"]
            paths += list((root/"tests/app/moire").glob("test*.py"))
            for path in sorted(paths):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                data = path.read_bytes()
                info = tarfile.TarInfo(str(path.relative_to(root)))
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                bundle.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["validate", "train", "revalidate", "revalidate_pilot"])
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    args = parser.parse_args()
    if args.mode in {"train", "revalidate", "revalidate_pilot"}:
        if args.validation is None:
            raise ValueError("Provide the actual GPU validation result.")
        gate = json.loads(args.validation.read_text())
        if not gate["passed"] or not gate["real_gpu"]:
            raise ValueError("Production requires a passing, real-GPU gate.")
    if args.mode.startswith("revalidate") and args.checkpoint_root is None:
        raise ValueError("Provide the mounted first-run checkpoint root.")
    root = Path(__file__).resolve().parents[3]
    args.out.mkdir(parents=True, exist_ok=True)
    payload = archive(root)
    digest = hashlib.sha256(payload).hexdigest()
    (args.out/f"source_{digest[:12]}.tar.gz").write_bytes(payload)
    encoded = base64.b64encode(payload).decode()
    data = yaml.safe_load(args.template.read_text())
    training = args.mode in {"train", "revalidate", "revalidate_pilot"}
    pilot = args.mode == "revalidate_pilot"
    label = {"train": "48h20-firstpoints", "validate": "1h20-validation",
             "revalidate": "48h20-convergence",
             "revalidate_pilot": "3h20-convergence-probe"}[args.mode]
    caption = f"jaqmc-mote2-cdw-R1-{label}-{digest[:8]}"
    data["caption"] = caption
    resource = copy.deepcopy(data["jobDefVersion"]["resource"])
    arnold = resource["arnoldConfig"]
    arnold["multipleMachine"] = training
    arnold["roles"][0].update(cpu=176 if training else 22,
        memory=1859584 if training else 232448, gpu=8 if training else 1,
        num=6 if training else 1, ports=1 if training else 0)
    if pilot:
        arnold["multipleMachine"] = False
        arnold["roles"][0].update(cpu=66, memory=697344, gpu=3, num=1, ports=0)
    arnold["hdfsVolumes"] = [dict(
        path="hdfs://haruna/home/byte_data_seed/ssd_hldy/user/zhangyan.31415",
        accessMode="RW", roles=[], mnt="/mnt/hdfs/cdw", extra="", preheatDir="")]
    for target in [data["jobDefVersion"], data["jobRunParams"]]:
        target["resource"] = copy.deepcopy(resource)
    data["jobRunParams"]["comment"] = caption
    data["jobRunParams"]["envsList"] = dict(JAX_PLATFORMS="cuda", JAX_ENABLE_X64="true",
        JAX_DEFAULT_MATMUL_PRECISION="highest", XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1",
        SKIP_MERLIN_OFFICIAL_INTERNAL_INSTALL="TRUE")
    example = '"$WORK_DIR/code/example/MoTe2/cdw_reproduction"'
    default_rank = "" if training and not pilot else "0"
    script = f"""set -euo pipefail
export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cuda
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8
export UV_INDEX_URL=https://bytedpypi.byted.org/simple
export UV_DEFAULT_INDEX=https://bytedpypi.byted.org/simple
export UV_HTTP_TIMEOUT=120
HOST_RANK="${{MLP_ROLE_INDEX:-${{MLP_WORKER_ID:-${{ARNOLD_WORKER_ID:-${{ARNOLD_ID:-${{RANK:-{default_rank}}}}}}}}}}}"
WORK_DIR="/tmp/mote2-cdw-{digest[:12]}-${{ARNOLD_TRIAL_ID:-manual}}-host$HOST_RANK"
REMOTE_DIR="/mnt/hdfs/cdw/jaqmcruns/mote2-cdw/{args.mode}_{digest[:12]}/trial_${{ARNOLD_TRIAL_ID:-manual}}/host$HOST_RANK"
mkdir -p "$WORK_DIR/code" "$WORK_DIR/results" "$REMOTE_DIR"
finish() {{
  code=$?
  trap - EXIT
  printf 'exit_code=%s\\n' "$code" > "$WORK_DIR/results/exit_status.log"
  cp -r "$WORK_DIR/results/." "$REMOTE_DIR/" || true
  exit "$code"
}}
trap finish EXIT
base64 -d > "$WORK_DIR/source.tar.gz" <<'SOURCE'
{encoded}
SOURCE
python3 - "$WORK_DIR/source.tar.gz" <<'PY'
import hashlib,pathlib,sys
assert hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest() == "{digest}"
print("SOURCE_VERIFIED {digest}",flush=True)
PY
tar -xzf "$WORK_DIR/source.tar.gz" -C "$WORK_DIR/code"
cp "$WORK_DIR/source.tar.gz" "$REMOTE_DIR/"
uv venv --python 3.12 "$WORK_DIR/venv"
uv pip install --python "$WORK_DIR/venv/bin/python" -r {example}/requirements.txt
DRIVER_LIB="$(ldconfig -p 2>/dev/null | awk '/libcuda.so.1/ && !found {{print $NF; found=1}}')"
if [[ -z "$DRIVER_LIB" ]]; then echo "CUDA driver unavailable" >&2; exit 3; fi
export LD_LIBRARY_PATH="$(dirname "$DRIVER_LIB"):/usr/local/nvidia/lib64:${{LD_LIBRARY_PATH:-}}"
export PYTHONPATH="$WORK_DIR/code/src"
cd "$WORK_DIR/code"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"""
    if args.mode.startswith("revalidate"):
        checkpoint = ("$CHECKPOINT_ROOT" if pilot else "$CHECKPOINT_ROOT/host$HOST_RANK")
        ranks = " --ranks 0,12,36,24" if pilot else ""
        script += f"""
if [[ ! "$HOST_RANK" =~ ^[0-5]$ ]]; then echo "Invalid host rank $HOST_RANK" >&2; exit 4; fi
CHECKPOINT_ROOT={shlex.quote(str(args.checkpoint_root))}
"$WORK_DIR/venv/bin/python" -u {example}/launch_revalidation.py --out "$WORK_DIR/results" --remote "$REMOTE_DIR" --host "$HOST_RANK" --checkpoint-root "{checkpoint}"{ranks}
"""
    elif training:
        script += f"""
if [[ ! "$HOST_RANK" =~ ^[0-5]$ ]]; then echo "Invalid host rank $HOST_RANK" >&2; exit 4; fi
"$WORK_DIR/venv/bin/python" -u {example}/launch.py --out "$WORK_DIR/results" --remote "$REMOTE_DIR" --host "$HOST_RANK" --steps 8000
"""
    else:
        script += f"""
"$WORK_DIR/venv/bin/python" -u {example}/run.py validate --out "$WORK_DIR/results" --width 32 --depth 3 --walkers 16 --steps 5 --sweeps 20 --cutoff 9 2>&1 | tee "$WORK_DIR/results/validation.log"
"""
    data["jobDefVersion"]["entrypointFullScript"] = script
    data["jobRunParams"]["entrypointFullScript"] = script
    data["jobDefVersion"]["name"] = caption
    data["jobDefVersion"]["gitRepo"] = {}
    data["jobDefVersion"]["scms"] = []
    destination = args.out/f"{args.mode}_{digest[:12]}.yaml"
    destination.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    (args.out/f"{args.mode}_provenance.json").write_text(json.dumps(dict(
        caption=caption, source_sha256=digest, archive_bytes=len(payload),
        yaml=str(destination), validation=str(args.validation) if args.validation else None,
        checkpoint_root=str(args.checkpoint_root) if args.checkpoint_root else None),
        indent=2))
    print(json.dumps(dict(yaml=str(destination), source_sha256=digest,
                          archive_bytes=len(payload)), indent=2))


if __name__ == "__main__":
    main()
