# 18-hole MoTe2 layer-sampler benchmark

Continuum bilayer MoTe2 at **2 degrees, 18 holes in 27 moire cells, nu=-2/3**,
physical-spin sector FM(18,0). This experiment trains a complex neural
wavefunction, freezes one checkpoint, and compares discrete Metropolis layer
flips, exact single-site Gibbs draws, proxy-corrected Wolff clusters and
continuous layer pseudospins. The pseudospin labels the two layers.

This is runnable experimental code, **not a completed efficiency result**.
Training convergence and sampler calibration must pass before reporting a win.
Filling and a numerical optimization plateau do not establish FCI topology.

## Environment

Use a dedicated Python **3.12** environment from this source checkout:

```bash
python3.12 -m venv .venv-mote2
source .venv-mote2/bin/activate
python -m pip install -r example/MoTe2/layer_sampler_benchmark/requirements.txt
python -m pip install 'jax[cuda12]==0.11.2'
python -c 'import jax; print(jax.__version__, jax.devices())'
```

The tested numerical runtime is Python3.12.7, JAX/JAXLIB0.11.2,
Flax0.12.10, NumPy2.5.3 and SciPy1.18.1. Use a JAX GPU build supported by the
target machine's driver; the CUDA12 command above names the build used here.
This example loads this checkout's `src/jaqmc` directly and needs no parent
research project, private dependency directory or editable installation.
The main project's package metadata currently caps JAX at0.9.1, so **do not
combine `pip install -e .` with this example's0.11.2 runtime**. Other JAX
versions have not been validated for this experiment. Kinetic energy uses
`scan` or `forward_laplacian`; the JAX0.11 gather API and stack rules are
fixed in this branch. The latter is faster on the tested18-hole fixture.

## KFAC and forward Laplacian

Install the tested KFAC environment, then select both backends explicitly:

```bash
python -m pip install -r example/MoTe2/layer_sampler_benchmark/requirements-kfac.txt
python -m pip install 'jax[cuda12]==0.11.2'
python example/MoTe2/layer_sampler_benchmark/run.py \
  --out "$PWD/runs/mote2_18h_kfac_forward" --gpus 0,1,2,3 \
  --optimizer kfac --kinetic-mode forward_laplacian --kfac-inverse-period 10
```

The optional requirements pin an upstream KFAC-JAX Git commit with JAX0.11
compatibility fixes, plus the tested Distrax/TFP dependencies. Installing the
published `kfac-jax==0.0.8` wheel alone is insufficient on JAX0.11.2.
The defaults remain `--optimizer sr --kinetic-mode scan` for reproducibility.
Both choices propagate through training, frozen validation and sampling.
KFAC initialization/compilation is recorded separately from hot-step timings.

KFAC uses the native complex-wavefunction implementation with real parameters,
variance1 for its registered predictive distribution, damping.05, EMA.95 and
one inverse refresh every10 steps by default. `--kfac-inverse-period 1` refreshes
every step. Its block approximation uses uncentered curvature; the SR backend
uses the centered empirical quantum metric. The directions need not be equal.
Both use the same force Re<conj(O)*(E_local-mean(E_local))> and the same centered
empirical quantum-metric/parameter trust caps. The force convention is half the
real energy gradient and is identical for both optimizers.

The float64 dense patterns include biases; the trainable momentum contraction
is exposed to graph matching. All parameters in the tested model occupy10
repeated-dense blocks. Unmatched `generic` shared-parameter blocks are rejected:
replicated-one-hot `fisher_exact` tangents can otherwise cancel their phase
curvature. There is an explicit rejection regression, not a silent fallback.

Training checkpoints retain KFAC EMA factors, inverse caches, optimizer count,
full sampler state and backend configuration. Resume requires the same config.
To deliberately migrate trusted weights from a previous optimizer/backend, add
`--reset-optimizer`; earlier holdouts are archived and new validation is needed.
Sampling continuation also requires the same kinetic backend so its timing
series is not mixed. A KFAC-trained checkpoint must be loaded in an environment
with its KFAC dependencies available.

[Measured18-hole checks and timings](BACKEND_VALIDATION.md) distinguish pure
local-energy evaluation, full optimizer updates and short actual training loops.
Single-step throughput does not establish time to a converged target energy.

To repeat those checks using a trusted trained128-walker checkpoint and its
matching Bloch input:

```bash
CUDA_VISIBLE_DEVICES=0 python example/MoTe2/layer_sampler_benchmark/benchmark_backends.py \
  --checkpoint /absolute/run/training/seed731/checkpoint_2000.pkl \
  --bloch /absolute/run/bloch_shell5.npz --out /absolute/backend_checks
```

## Run on another GPU machine

From the repository root, choose an absolute output directory:

```bash
python example/MoTe2/layer_sampler_benchmark/run.py \
  --out "$PWD/runs/mote2_18h_2deg" --gpus 0,1,2,3
```

For one GPU, use `--gpus 0`. Four independent logical replicas run in rounds
on1–4 selected GPUs, with one process per GPU. Training, fresh holdouts and
Wolff tuning use all selected GPUs where independent cases are available.
This is independent-replica parallelism, not one wavefunction distributed
across GPUs. IDs/UUIDs must refer to devices authorized and visible on that
machine. Keep float64 enabled; hardware with stronger float64 throughput is
useful for this workload. The tested128-walker microbatched training fits
one24GB RTX4090; a larger GPU is not required for correctness.

For a batch cluster, invoke this same command inside a GPU allocation and
pass the allocated visible devices. The controller remains in the foreground;
do not depend on a disconnected interactive shell keeping it alive.

Useful individual stages:

```bash
python example/MoTe2/layer_sampler_benchmark/run.py --out "$PWD/runs/mote2_18h_2deg" --gpus 0 --stage init
python example/MoTe2/layer_sampler_benchmark/run.py --out "$PWD/runs/mote2_18h_2deg" --gpus 0 --stage validate
# Other stages: train, evaluate, pilot, compare, analyze
```

Run stages in that order. `train` resumes each replica's `latest.pkl`.
`compare` skips already completed cases and can append samples from saved
positions, internal variables, RNG and orbital cache. Data and previous
measurement times are retained. Source and checkpoint hashes must match for
sampling continuation; do not edit the source during a run.
Outputs contain absolute artifact paths, so keep the output directory in place
during a run. Loading checkpoints requires trusted, task-owned pickle files.
No trained weights, local logs, machine allocations or research data are
included in this commit.

## What the controller does

1. Build full-continuum Bloch initialization orbitals (shell5), and check
   shell6, antisymmetry, Gamma momentum projection and Fourier/energy identities.
2. Train four independent128-walker SR/KFAC replicas:32x3 single-particle and16x3
   pair streams,8854 parameters. From-scratch strict training starts with half
   the base learning-rate schedule; restore-point strict holdouts are added
   when resuming coarse training. A fixed step count does not certify convergence.
3. Require at least8000 steps, two strict2048-record frozen holdouts,
   energy difference+1.96 combinedSE below0.02meV/hole, stable variance and
   mixing/drift checks. Cap16000 gives an incomplete result when gates fail.
   At least two of four independent replicas must pass; all failures are reported.
4. Re-evaluate candidates independently, select by the earlier holdout,
   validate that fixed choice, and freeze one file/SHA256 for every sampler.
5. Tune the Wolff proxy on12 independent pilot cases: coupling0/.15/.35/.7,
   radius.4/.65/1 moire units. Always retain exact Metropolis correction.
6. Run8 seeds x32 walkers x4096 records/method, cadence10, burn4100 sweeps;
   fine-cadence checks and separate Gibbs/continuous references follow the
   fixed schedule. Rotate methods over four logical GPU slots.
7. Compare measured GPU seconds, independent-walker/block errors and a separate
   reference-based95% absolute-error bound. Report observed precision crossings
   at.02/.01/.005/.002meV/hole. GPU seconds sum sampler+energy work across chains;
   they are distinct from the controller's wall-clock time.
8. Only scientific-calibration exit3 triggers exact continuation to8192 main
   and16384 reference records. Other errors stop immediately. Continued failure
   preserves diagnostics and exits without certifying an efficiency ranking.

The continuum Hamiltonian retains layer potentials, tunneling, valley kinetic
shifts and coplanar2D Ewald interactions: V=11.2meV, w=-13.3meV, phi=-91deg,
m*=.62, epsilon=10. Bloch orbitals are an initialization, not a production
band projection. The architecture uses primitive-periodic one-body inputs and
supercell-periodic relative inputs, enabling exact reuse of the neural core
across the27 translations of the same Gamma projection. This differs from
an unrestricted native supercell-feature architecture. The common scalar
Jastrow removes the unlike-layer s-wave Coulomb pole; it does not enforce
every angular-momentum cusp. Slow higher-moment tails remain possible.

## Outputs

- `CURRENT_STAGE.txt`, `logs/`: controller stage and per-process logs.
- `training/seed*/`: full-state checkpoints, frozen traces, gate outcomes.
- `frozen/selection.json`: all replica statuses, physical configuration,
  Bloch hash, independent evaluations and common checkpoint hash.
- `pilot/`, `production/`: raw traces, sampler states and measured timings.
- `REPORT_ZH.md`, `energy_precision_vs_time.png`, `SCIENTIFIC_STATUS.json`:
  final report, figure and explicit scientific acceptance status.

A report can exist while scientific acceptance is false. Inspect
`SCIENTIFIC_STATUS.json`; unsuccessful mixing/reference/tail checks suppress
claims. Precision crossings are retrospective diagnostics, not guarantees
under optional stopping. Compiling and extra correlation observables are
recorded separately from the sampler+energy timing.

## Verification performed for this handoff

The portable copy was launched from a separate working directory on a24GB
RTX4090: continuum initialization, the real18-particle physical identities,
20 strict-training steps and all four sampler smoke cases passed. Fourier
reconstruction errors were below4.2e-12meV/hole and the shell5-to6 band
change was below1.42e-7meV. These checks establish the tested execution path;
they do not establish long-training convergence, full-run performance, or
installation success on an untested driver. The example-local environment
requirements describe the dependencies used, rather than a new full-system
lockfile.

Two CPU numerical-contract tests compare sample-space SR against an independent
parameter-space metric solve and check microbatch output ordering:

```bash
JAX_PLATFORMS=cpu python -m unittest discover \
  -s example/MoTe2/layer_sampler_benchmark -p test_helpers.py -v
```
