# Original AB Γ VBM: same-model L-BFGS-B continuation

Exactly the same10 physical coefficients, selected potential indices[7,0], canonical reference, native residual and bounds were used. The warm snapshot is saved with its SHA256. Only numerical optimization changed: one fixed initial-Jacobian column scaling followed by L-BFGS-B.

The optimizer genuinely stopped on projected-gradient tolerance after99function evaluations and79iterations, reducing loss106.7905121→91.0926579. Final pair-centre RMS is0.658294meV, full six-state RMS1.305060meV. This rules out interpreting the slow preceding Gauss-Newton trajectory as evidence that this family had already saturated.

warm_snapshot: Pmin=[0.7323277860665806, 0.34348986910852064, 0.2544789436931395], PPmin=[0.7076755802828368, 0.005940394503574612, 0.004412639057311696], centres=0.712746meV, six=1.54094meV.

final_lbfgs: Pmin=[0.7238190946119054, 0.45145692792025754, 0.34363328494720913], PPmin=[0.7060563964476136, 0.19383737561732015, 0.15524340012508614], centres=0.658294meV, six=1.30506meV.

Physical principal lower coefficient is1.16223526eV Å². Fresh initial/final directional FD errors are3.65e-09/1.79e-09; scaled projected half-loss gradient infinity norm is6.01e-07, unscaled projected norm2.66e-05. Delta_lower coordinate hits its existing lower bound z3=−7, corresponding to1.0182376meV. The outward bound gradient is permitted by KKT; it was not mistaken for free stationarity. No bound relaxation was run.

Q37→61→91 maximum absolute first6 changes are.403371/.028817meV, centred changes.349547/.026378meV. This is model-Q sensitivity against unchanged native sourceQ, not source-Q convergence. Every final P/PP is raw and unnormalized. Fresh six-k diagonalization reproduces saved energy/P/PP to floating precision.

BANDS_OVERLAP.png gives both centre and full-six RMS; solid overlap lines are P, dashed are worst-principal PP. This remains a single-angle numerical control with modest overlaps, not a qualified all-angle NQS model or a proof that the family is impossible. The active bound and possible other basins remain explicit. Runtime gpuh201, /data/home/zy/mambaforge/envs/moirekp Python3.11.15,4k workers/BLAS1. Main process and independent verification exited0.
