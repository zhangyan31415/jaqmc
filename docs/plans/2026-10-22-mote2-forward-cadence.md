# Frozen MoTe2 forward-Laplacian measurement cadence

User-authorized experiment: keep checkpoint SHA256
6f79f2570485b3202cf40fac72c20b230ca286108e1faed5c0bb63a44f6b2423
fixed, measure discrete flip, conditional, corrected Wolff and continuous
sampling at 5/10/20/40 sweeps using the repaired forward Laplacian.

Runtime: /data/work/zy/software/research-lab/10_nqs/.venv, float64,
32 independent walkers per seed, microbatch 8. Keep spatial width .07,
continuous width pi, Wolff coupling .35/radius 1, and 4100 burn sweeps.
No neural optimization and no retuning of other sampler parameters.

1. Verify forward-vs-scan complex energies on saved-chain configurations of
   all four methods from the exact final checkpoint. Fail before sampling
   if numerical agreement, source hashes or finite values fail.
2. Pilot all 16 combinations: 8 seeds x 32 walkers x 2048 records per case.
   Use distinct seeds from the previous study. Rotate combinations across
   allocated GPUs and execution order to distribute hardware/time effects.
3. Select each method's cadence by measured (sampling + energy) GPU time
   times independent-walker MCSE squared. Report bootstrap cadence-selection
   frequencies and pairwise intervals; do not claim a unique optimum when
   uncertainty leaves alternatives unresolved.
4. Freeze selection.json before a new 12-seed x 32-walker x 4096-record
   holdout. Run each selected method/cadence and a cadence-10 baseline when
   different. These seeds are independent of the pilot and prior study.
5. Report MCSE, independent-walker and block/spectral checks, energy/layer/
   charge Rhat, initial-family checks and mean agreement. Primary uncertainty
   resamples whole independent walkers stratified by seed, preserving all
   time correlation. Resample matching walker pairs across compared cases.
6. Report actual measured cumulative GPU seconds at stable observed MCSE
   thresholds .02/.01/.005/.002, with/without burn. Bootstrap the observed
   crossing distribution with explicit right-censoring when a replicate
   does not cross. These are retrospective estimates, not confidence
   sequences or optional-stopping certificates. Also report asymptotic
   cost/precision estimates, labeled extrapolations, with 95% intervals.
7. Keep common validation, compile, pilot tuning, I/O and final diagnostic
   time distinct from production timing. Do not call GPU-sum time wall time
   or MCSE an exact-ground-state error. Preserve all traces and provenance.

Implementation: add cadence statistics contracts first, then experiment
worker, selector/analyzer and Slurm runner. Reuse benchmark_worker.run_case
and existing physical kernels; snapshot exact package sources before launch.
Verify contracts, a GPU smoke run and final artifacts before reporting.
