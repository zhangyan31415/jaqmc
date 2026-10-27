# Optional optimized moiré workflows

This example packages the measured MoTe₂ and Γ-valley optimization paths in
`jaqmc.app.moire.optimized`. The existing JaQMC workflow remains unchanged until
the optional backend is selected. Network sizes, full Hamiltonians, and native
optimizer/checkpoint structures are retained.

## Run

Use an isolated Python 3.12 environment with this checkout's `src` on the import
path and the dependencies installed. GPU commands should run inside your normal
compute allocation. The commands below run from the repository root:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# MoTe2: full N18, filling 1, FM, theta=3 degrees; continuous auxiliary angles.
python example/moire_optimized/run.py \
  --model mote2 --out "$PWD/runs/mote2-continuous" --steps 1000 \
  --batch 512 --chunk 512 --burn 1000 --seed 731 \
  --representation continuous --derivative sparse-lap

# The same layer basis represented by binary labels, with exact layer Gibbs updates.
python example/moire_optimized/run.py \
  --model mote2 --out "$PWD/runs/mote2-discrete" --steps 1000 \
  --batch 512 --chunk 512 --burn 1000 --seed 731 \
  --representation discrete --derivative sparse-lap

# AB-ZrS2 six-parameter Gamma model: N18, two model-spin sectors 9+9.
python example/moire_optimized/run.py \
  --model gamma --out "$PWD/runs/zrs2-gamma" --steps 1000 \
  --batch 512 --chunk 512 --burn 1000 --seed 731 \
  --derivative jvp --directions 6
```

`--steps` means **additional optimizer updates**, including when resuming.
Use a new output directory for each invocation. `--resume` restores a compatible
native checkpoint and continues its update counters and random-number state;
it is not a parameter-only transfer. For example:

```bash
python example/moire_optimized/run.py \
  --model mote2 --out "$PWD/runs/mote2-continuation" --steps 1000 \
  --batch 512 --chunk 512 --burn 1000 --seed 731 \
  --representation continuous --derivative sparse-lap \
  --resume "$PWD/runs/mote2-continuous/train_ckpt_000999.npz"
```

The native checkpoint filename stores a zero-based completed step: step `999`
corresponds to 1,000 completed updates. Keep the same physical configuration,
representation, batch, and optimizer settings on continuation. Do not resume
continuous angles as binary layer labels, or binary labels as angles.
`--checkpoint-every 5000` controls intermediate saves; the final state is saved
also. `--config path/to/config.json` supplies a native configuration instead of
the bundled material preset. See `--help` for supported combinations and options.

## Which model and representation?

| Preset | Physical model | Internal components | Default derivative |
| --- | --- | --- | --- |
| `mote2` | Twisted bilayer MoTe₂, 3°, 18 holes in 18 moiré cells, filling 1, FM `(18,0)` | Two layers | `sparse-lap` |
| `gamma` | AB twisted bilayer ZrS₂, 3.15°, six-parameter Γ model, 18 particles, model spins `(9,9)` | Layer × orbital = four per spin sector | Six-direction `jvp` |

Both presets use one determinant and the full FermiNet architecture with four
256-wide single-stream layers and four 32-wide pair-stream layers. The Γ
operator includes both `a1` scalar and `a2` matrix kinetic terms, including
mixed derivatives, the full matrix potential, and native Ewald interactions.
Its four internal components are **not four physical layers**. The portable
Γ parameters and reciprocal vectors are bundled in
[`models/zrs2_gamma_3p15.json`](models/zrs2_gamma_3p15.json); no external fitting
repository or machine-specific model path is required. This is the six-parameter
real-space model, not a projected four-band Hamiltonian.

For MoTe₂ the continuous representation mixes layer orbitals with complex
weights `exp(+is)` and `exp(-is)`. The discrete representation selects layer
0 or 1 using a one-hot vector. The neural layer orbitals themselves are the
same variational family. Binary labels are sampled, not held fixed; interlayer
tunnelling remains in the energy. These variables encode **layer pseudospin**,
not a change in the physical spin/valley particle counts.

The continuous sampler retains 20 native position/angle Metropolis steps. The
discrete sampler uses 20 coordinate Metropolis steps followed by a complete
layer Gibbs sweep, reusing layer orbitals and determinant updates where valid.
The tested discrete version was slightly slower overall; it is an optional
representation, not an advertised performance improvement.

## Backend choices and boundaries

MoTe₂ merges kinetic and valley SOC derivatives through a covariant phase,
uses weighted covariance gradients, and selects Cholesky KFAC. Sparse Forward
Laplacian preserves the actual local derivative support of pair features;
global single-particle dependencies remain present. The current sparse/discrete
adapter validates bilayer, single-phase, one-determinant, `chunk=512` inputs.
For MoTe₂, `--derivative baseline` and `--derivative lap` are aliases for this
same **dense merged** Forward Laplacian backend. Neither selects the historical
unoptimized MoTe₂ implementation; `sparse-lap` selects sparse propagation.

Γ supports `baseline`, `jvp`, `lap`, and `sparse-lap`. They compute the same
complete differential operator with different directional propagation paths;
they are alternatives and their speedups cannot be multiplied. `baseline`
means the **already merged** sequential derivative implementation, with
weighted gradients and Cholesky KFAC still enabled. It does not mean the
historical, unoptimized MoTe₂ timing column called “original”.

Representation selection changes process-local wavefunction dispatch. Construct
and trace one representation per process. Run the continuous and discrete
examples as separate commands, as shown above.

Historical measurements, actual numerical discrepancies, and convergence limits
are recorded in [BENCHMARKS.md](BENCHMARKS.md). The Γ candidates are included by
explicit user choice despite their recorded maximum-sample FP32 energy gate
failures. Inclusion does not turn those historical checks into passes.

## Environment and verification

[requirements-tested.txt](requirements-tested.txt) records the scientific
packages in the environment actually used for the historical calculations.
That manually assembled environment used JAX/JAXlib 0.11.2, which is **outside
the current `pyproject.toml` JAX upper bound of 0.9.1**. The manifest is an
experimental environment record, not a claim that a standard dependency
resolver can install it together with the unmodified project constraints.
Package-wide compatibility bounds are not changed by this example. Install
into an isolated environment, not a shared base installation.

The packaged regression tests live in `tests/app/moire/optimized`. They check
mathematical identities and portable APIs. Historical GPU timings are archived
measurements, not newly measured by a CPU test run. This repository example does
not include trained weights, long trajectories, compilation caches, or the
stopped long-training study.

For a reduced CPU check of the actual workflows and full-state continuation:

```bash
JAX_PLATFORMS=cpu python example/moire_optimized/smoke_test.py \
  --out "$PWD/runs/optimized-cpu-smoke"
```

This uses two particles, batch4 and small networks. It runs continuous MoTe₂,
binary-layer MoTe₂ and Γ workflows, and compares uninterrupted3 updates against
1+2 resumed updates, including every native state array and PRNG. It is a
software smoke test, not a material result or a GPU performance measurement.
