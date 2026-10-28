# AA 2H-WTe2 3.89-degree K-valley model audit

## Status

The symmetry-adapted 54-dimensional projected reference is retained as the
highest-fidelity reduced representation. No four- or six-parameter continuum
model is released because every compact candidate fails the wavefunction
overlap gate.

This dataset describes trigonal-prismatic, 2H-class WTe2. It must not be
presented as a model of the experimentally common distorted 1T'/Td WTe2 phase.

## Scope and reference validation

- Structure: AA bilayer 2H-WTe2
- Twist angle: 3.89 degrees
- Valley: K1
- Reference path: 61 points on G-M-K-G
- TAPW projection cutoff: Q-shell 6
- Projected model size: 54 x 54 at each path point

The regenerated TAPW source agrees with the legacy top-three bands with
0.034 meV RMS and 0.052 meV maximum error, using one stable band-index pattern.
The symmetry-adapted projected top-three bands agree with the native reference
to 0.360 meV RMS after a common energy shift.

C3z and C2T were accepted as approximate source symmetries only after setting
the explicit tolerance to 0.02; the measured C3 subspace leakage was 0.01735.
The basis reports validated symmetry closure, while the selection artifact
still records `PENDING_SYMMETRY`, so this is a reference-quality result rather
than a fully certified release.

Reference location:

`kp_reference_symmetry/K/K1/q06/projection/`

## Source-band topology

The independent Q-shell-7 TAPW packet gives:

| Band | Chern number |
|---|---:|
| -1 | 0.999987 |
| -2 | 0.999954 |
| -3 | -1.999332 |
| -4 | 0.999281 |

The highest valence band has a 43.887 meV bandwidth and a 17.141 meV minimum
direct gap to the next band. These Chern numbers belong to the source TAPW
bands; they were not transferred to any rejected compact model.

## Compact-model audit

The acceptance rule required a minimum path wavefunction overlap of at least
0.93. None of the 20 four-parameter or 30 six-parameter trials passed.

| Candidate selected by its development objective | Path RMS / meV | Path max / meV | Mean overlap | Minimum overlap |
|---|---:|---:|---:|---:|
| four parameter | 4.277 | 14.524 | 0.2000 | 0.0236 |
| six parameter: second harmonic | 3.609 | 7.430 | 0.2031 | 0.0156 |
| six parameter: momentum tunneling | 3.353 | 8.877 | 0.2086 | 0.0187 |

The best energy-only validation values were 2.744 meV for the second-harmonic
family and 2.913 meV for momentum tunneling. The 2.744 meV trial reached the
iteration limit, and both retained negligible wavefunction overlap. Energy
agreement alone is therefore not sufficient.

Diagnostic parameter sets, not approved for physics calculations:

| Family | m_h/m_e | V1 / meV | phi1 / deg | w1 / meV | Extra terms |
|---|---:|---:|---:|---:|---|
| four parameter | 0.3499 | 27.513 | -180.000 | -31.441 | none |
| second harmonic | 0.2804 | 36.994 | 68.217 | -6.829 | V2=-80.000 meV, w2=26.476 meV |
| momentum tunneling | 0.3487 | 16.846 | 157.087 | -17.712 | beta=(-1.5900-0.0300i) eV A^2 |

The second-harmonic `V2` value is at the imposed lower bound, which is an
additional sign that this family is compensating for missing physics.

Rejected fit artifacts are stored under:

- `fits/no_symmetry/`
- `fits/symmetry_reference/`

## Interpretation and next model

The reliable conclusion is that the native WTe2 top-valence subspace exists
and is topological, but a scalar one-orbital-per-layer MoTe2-style Hamiltonian
does not reproduce its wavefunctions. A next-generation compact model should
keep at least a two-orbital internal space per layer, derive its operator basis
from the 54-dimensional symmetry-adapted Heff, and validate topology from the
fitted eigenvectors before use in many-body calculations.
