# Portable moire optimization implementation plan

**Goal:** Package the existing MoTe2 and Gamma-valley implementations as reusable optional backends, verify them against independent numerical fixtures, commit and push to `zy/mote2-layer-sampler-benchmark`.

**Architecture:** Add `jaqmc.app.moire.optimized` without changing the existing default workflow. Share weighted covariance gradients and KFAC settings. Provide continuous/discrete bilayer adapters and complete four-component Gamma operators with baseline, grouped JVP, Forward Laplacian and sparse options. Replace research-directory imports and model paths with package imports and a bundled model JSON. User approved including Gamma candidates despite the historical maximum-energy gate; preserve measured errors transparently.

**Tech stack:** Existing Python3.12/JAX/JaQMC/Flax/KFAC; existing project .venv for verification. No new long physics calculations, cached weights, HDF5 trajectories or compilation caches in Git.

## Tasks and ownership

1. MoTe2/shared agent: port `shared.py`, `covariant.py`, `folx_compat.py`, `mote2.py`, `discrete.py`, `sampling.py`; add focused tests of covariance, representation, sampler and derivative identities. Preserve complete Hamiltonian and original parameter shapes.
2. Gamma agent: port `gamma.py`, `gamma_operator.py`, `gamma_directions.py`, `gamma_model.py`; bundle the frozen six-parameter AB-ZrS2 model at3.15degrees; independent determinant-Hessian tests, model-loader validation and no absolute paths.
3. Root: package exports, portable example CLI/configs, full native checkpoint/PRNG continuation, docs and concise measured benchmark records; actual source imports must resolve inside this checkout.
4. Independent review: inspect public APIs, physical terms, portability, state continuity and benchmark scope; fix material findings.
5. Verification: focused CPU tests and CLI smoke from a directory outside the checkout with only repository source/dependencies exposed; no research-path imports. Check staged content for large or private files, commit selected paths and push existing SSH origin branch. Verify remote commit matches local HEAD.

## Verification requirements

- Red/green tests for newly exposed APIs before implementation where practical; port existing verified mathematical fixtures rather than implementation-mirroring tests.
- Shared gradients match explicit centered covariance; both layer representations retain all physical operators; Gamma retains a1/a2 and mixed derivatives, four internal components and backflow.
- Document actual tested environment and the current package JAX version constraints; do not silently change global dependency bounds.
- Historical Gamma201→158ms measurements remain same-chunk paired timings, not proof of equal-convergence speed. No tests pretend the historical numerical gate passed.
- Original research artifacts and the stopped70k study remain untouched.
