# WS2 9.43-degree K1 Q3 projection report

## Status

The 24-dimensional projected Hamiltonian passes the full 61-point acceptance
test for the highest three valence bands.

The accepted scope is the top three bands only. The result must not be
described as an accurate top-eight-band model.

## Provenance

- TAPW input: `ws2_9.43_q03_current_root_bigmem001/K1/q03/band`
- TAPW source commit: `3b6ca7f93e07d1e12e35d91c228476acce856712`
- Material: AA bilayer WS2
- Twist angle: 9.430008 degrees
- Valley: K1
- Q-shell: 3, 12 Q vectors per layer
- k path: G-M-K-G, 61 points
- Downfolding: energy-linearized Lowdin
- Reference energy: -5.82256669559 eV

The input TAPW calculation was independently reproduced on bigmem001 and
bigmem003 before projection. The full 4.55 GB H(k) files were bitwise
identical across nodes.

## Source symmetry

- C3z raw-H covariance residual: 6.64248e-4
- C2T raw-H covariance residual: 6.64090e-4
- C2T square residual: 7.21e-17
- C2T G-permutation maximum error: 1.26e-10 inverse Angstrom
- C2T operation gauge: W = -120 degrees, S = 0 degrees

The original C2T transport omitted this operation-specific relative species
phase. Its same-species blocks passed, while the W-S and S-W blocks produced
a residual near sqrt(3), causing a total covariance residual of 1.049. The
configured species phase restores C2T covariance without changing geometry,
unitarity, or C2T squared.

The phase is now derived from the source Fourier projector and atomic C2T
mapping in `c2t_derivation/DERIVATION.md`. It comes from the nonzero reciprocal
folding vector `B = K_target + R K_source` for the selected 0-degree C2 axis.
It is not an element-specific property of W. The accepted MoTe2 comparison
uses a 60-degree axis with B=0; forcing its 0-degree axis reproduces the same
missing-phase failure (0.851 before, 1.99e-4 after the geometric correction).
WS2's 60-degree axis also passes raw-H covariance without a species correction
(7.51e-6), but its 5.10e-8 inverse-Angstrom G-map error exceeds the current
1e-8 axis-construction gate. Production outputs here still use the configured
0-degree-axis action; this derivation audit does not change that action.

## Projected symmetry certification

- C3z unitarity residual: 1.11e-16
- C2T unitarity residual: 0
- C3z cubed plus identity residual: 2.48e-16
- C2T squared minus identity residual: 0
- C2T-C3z dihedral relation residual: 1.11e-16

Both C3z and C2T are exported in the certified 24-dimensional
`representations.npz` package.

The Q-coordinate canonicalizer uses an explicit relative cleanup tolerance of
3e-8. The maximum correction is 1.47800e-8 inverse Angstrom on a Q scale of
0.580078 inverse Angstrom. Point ordering is unchanged, and the maximum
symmetry-closure error decreases from 2.55456e-8 to 1.89e-16 inverse
Angstrom. The constrained-solver residual is 5.90e-16, below its certified
backward-error bound of 1.01e-13.

## Selected subspace

| Physical layer | Local band | Gauge anchor row |
|---|---:|---:|
| 0 | 11 | 24 |
| 1 | 11 | 23 |

- Model dimension: 12 Q x 2 layers = 24
- Both physical layers active: yes
- Basis orthogonality maximum residual: 1.78e-15
- Automatic-selection preview RMS / maximum: 0.902 / 2.345 meV
- Gauge anchor sigma minimum: 0.496

No inactive-layer diagnostic waiver was used.

## Full-path validation

### Energy, top three bands

| Metric | Value |
|---|---:|
| Absolute RMS | 0.937 meV |
| Absolute maximum | 2.034 meV |
| Gamma-aligned RMS | 1.474 meV |
| Gamma-aligned maximum | 3.107 meV |

The largest absolute error is the highest band at k index 40 (K).

Minimum adjacent direct gaps:

| Gap | Native TAPW | Projected Heff |
|---|---:|---:|
| band -3 to -2 | 3.129 meV | 2.780 meV |
| band -2 to -1 | 12.383 meV | 12.348 meV |

### Wavefunctions, top three bands

| Metric | Mean | Minimum |
|---|---:|---:|
| Individual band overlap | 0.9932 | 0.9813 |
| Native-state capture by the 24D subspace | 0.9933 | 0.9816 |
| Native spin-up amplitude | 0.9990 | 0.9946 |

The minimum overlap occurs for band -3 at k index 15.

## Acceptance

The result passes all configured hard gates:

- absolute energy RMS <= 2 meV
- absolute energy maximum <= 5 meV
- individual overlap minimum >= 0.98
- subspace capture minimum >= 0.98
- native spin-up amplitude minimum >= 0.99
- both layers active
- source C3z/C2T covariance residual <= 0.02
- projected group-relation residual <= 1e-10
- Q canonicalization certified with unchanged ordering

Machine-readable evidence: `PROJECTION_VALIDATION.json`.

## Artifacts

- Configuration: `kp_reference_symmetry.yaml`
- Projected Hamiltonian:
  `kp_reference_symmetry/K/K1/q03/projection/heff.npy`
- Basis:
  `kp_reference_symmetry/K/K1/q03/projection/basis.npz`
- Band comparison:
  `kp_reference_symmetry/K/K1/q03/projection/band_comparison.pdf`
- Source symmetry package:
  `../ws2_9.43_q03_current_root_bigmem001/K1/q03/symmetry/representations.npz`
- Projected symmetry package:
  `kp_reference_symmetry/K/K1/q03/symmetry/representations.npz`
