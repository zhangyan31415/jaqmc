# ZrS2 AB six-coefficient angle experiment

All models use 19Q, two molecular orbitals per layer and fixed AB phases. Energy, projector, direct and sampled indirect-gap weights are all one. Sources and Schur reference energies are recorded per case. Successful solver termination and model accuracy are reported separately.

| Angle | Solver | Dev gate | Path RMS / meV | Path min overlap | Direct gap / meV | Sampled indirect / meV | Fresh local gap RMS / meV | Fresh 2D RMS / meV | Fresh min native overlap |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 9.43 | converged | False | 17.272 | 0.0070 | 3.711 | -169.597 | 1.292 | 15.517 | 0.0466 |
| 7.34 | converged | False | 8.859 | 0.0152 | 1.504 | -126.131 | 1.967 | 8.793 | 0.0094 |
| 6.01 | converged | False | 3.238 | 0.8141 | 2.935 | -81.850 | 0.951 | 1.991 | 0.8696 |
| 5.09 | converged | False | 2.220 | 0.9259 | 3.462 | -47.362 | 1.026 | 1.677 | 0.9256 |
| 4.41 | converged | False | 1.816 | 0.9456 | 3.414 | -23.619 | 0.930 | 1.156 | 0.9415 |
| 3.89 | converged | False | 1.578 | 0.9427 | 3.047 | -8.770 | 1.583 | 1.034 | 0.9377 |
| 3.48 | converged | False | 1.511 | 0.9400 | 2.846 | -1.669 | 1.939 | 1.402 | 0.9314 |
| 3.15 | converged | True | 1.637 | 0.9317 | 1.885 | 1.885 | 1.468 | 1.820 | 0.9305 |
| 2.88 | converged | False | 1.587 | 0.9212 | 1.845 | 1.845 | 0.731 | 1.673 | 0.9382 |
| 2.65 | converged | False | 1.436 | 0.8556 | 1.304 | 1.304 | 0.974 | 1.009 | 0.9253 |
| 2.45 | converged | False | 1.299 | 0.6325 | 0.545 | 0.397 | 0.307 | 0.737 | 0.8608 |
| 2.28 | converged | False | 1.164 | 0.7885 | 3.374 | 0.229 | 0.039 | 1.108 | 0.8789 |

The development gate copies the 3.15 thresholds; it is not a universal physical validity test. In particular an intrinsically negative native indirect gap cannot satisfy the copied positive-gap threshold. Native direct and indirect gaps and error decomposition are included in summary.csv/json.

Fresh validation comprises six local and six 2D native points generated only after model freeze. These are finite sampled diagnostics, not full-BZ guarantees. The sweep transports the fixed 3.15 atomic anchor without a per-angle exact symmetry certification; 3.15 and 2.88 use existing pre-symmetrized real-space inputs, other angles use raw SOC inputs with C3 averaging. Reference-to-native errors must be inspected before attributing errors solely to the six-parameter family.
