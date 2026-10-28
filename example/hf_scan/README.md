# Continuum models for an external HF scan

This directory indexes the existing single-particle continuum models. It does
not implement Hartree-Fock, select an interaction, or certify a phase diagram.
Models and reports are grouped by material, stacking, valley and version under
`example/`. Historical models remain alongside updated candidates.

## Find and inspect a model

From the repository root, using Python and the selected package's NumPy/SciPy
dependencies:

```sh
python example/hf_scan/catalog.py list
python example/hf_scan/catalog.py list --material CdCl2 --json
python example/hf_scan/catalog.py show CdCl2_AA_Gamma_scalar3_schur
```

`MODEL_INFO.json` records the target band edge, physical parameter count,
global angle-law coefficient count, calibrated geometries, carrier/energy
conventions, basis and scientific limitations. `status` and `hf_scope` are
part of the input definition: a converged fit is not necessarily accurate.
The dated catalogue includes both candidate and failed/diagnostic versions.
List everything first; choose an explicit model ID rather than silently
selecting whichever file has the latest timestamp.

## Export the one-body matrix

Packages with `hf_api` have a uniform adapter:

```sh
python example/hf_scan/catalog.py export CdCl2_AA_Gamma_scalar3_schur --angle 3.890238169 \
  --k 0 0 --k 0.25 0.125 --eigenvalues --output one_body.npz
```

Alternatively `--nk 4` exports a uniform fractional grid in `[0,1)^2`.
The output stores `H`, `B`, `k_fractional`, `k_cartesian`, `theta_deg`, and a
JSON metadata string; `--eigenvalues` adds the unshifted eigenvalues.
`k_cartesian = k_fractional @ B`, with reciprocal vectors as rows in inverse
Angstrom and energy in eV. The **carrier convention is explicit**: a hole
Hamiltonian is not silently converted into an electron Hamiltonian.
Use a catalogued geometry angle. Angle-law interpolation does not by itself
validate a new geometry or small-angle extrapolation.

Dense exports are limited to 512 MiB and existing output files are preserved.
For production grids, export batches or call each package's documented matrix
API directly. Keep different packages in separate processes; portable modules
may use identical names such as `model` and `kernel`.

## Inputs the HF implementation must decide

- Interaction convention: dielectric environment, gate geometry, Coulomb
  kernel, neutralizing background, Hartree subtraction and exchange cutoff.
- Filling, active valleys/physical spins, temperature or smearing, allowed
  broken symmetries, initial states and convergence criteria.
- Density vertex in the archived envelope basis. Do not assume an arbitrary
  downfolded/source-dressed orbital is an atomic orbital or that the projected
  density is the identity. Retaining all internal components and projecting
  to bands require different form factors; source wavefunctions are not
  automatically supplied by this lightweight one-body archive.
- Matched plane-wave order, reciprocal lattice, energy/carrier signs and
  boundary conditions. In the real M-valley gauge the half-reciprocal
  tunnelling requires an area-two periodic cell or layer-twisted primitive
  boundaries; identical primitive scalar boundaries are incorrect.
- Convergence of both the HF k mesh and one-body plane-wave cutoff. Existing
  model-Q checks are not convergence checks of the atomic/TAPW reference.

Report partial ranges explicitly: for example CdCl2 is strongest for the first
band, several M models for the first two pairs at small angles, and current
PtSe2/MgBr2 valence candidates remain diagnostic. Their inclusion allows a
reproducible comparison; it is not an instruction to treat them as validated
production inputs. The WSe2/MoTe2 model/Heff overlaps must not be described as
direct native-TAPW wavefunction overlaps.

The runnable NQS backends on a separate benchmark branch are a different
deliverable. Uploading these one-body models does not add every required
JAQMC many-body estimator or a trained ground-state checkpoint.
