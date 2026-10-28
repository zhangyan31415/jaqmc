# Separate AB Γ conduction pilot: one scalar field, minimal three-term candidate

**This is a different target from the original AB Γ valence request. It does not repair or replace that valence model.** All results here are at3.890238° only, with the actual configured viewer Q37. There is no all-angle joint fit, no parameter curve, and no JAQMC integration claim.

The source audit independently establishes a single bilayer-distributed bonding conduction Kramers doublet44/45 (not one orbital on each layer). The remote conduction partner is~.54eV away. NativeΓ/M/K vectors are freshly rebuilt; remaining57 k points use only this target's original conduction cache. Rank2 pair spans are independently orthonormalized; cross-pair Gram error after this is<7.1e-8. No source C normalization or model-aligned wave rotation is used. One fixedΓ-built source Bloch-composite frame is reused over k, so its minimum capture is .97791/.97706/.97757 and scalar upper bound is .95743/.95627/.83465; this approximate source embedding is a separate, quantified limitation.

## Results

|explicit model|three-centre RMS meV|raw pair P minima1/2/3|raw pair P means1/2/3|K pair-centre gap meV|
|---|---:|---|---|---:|
|3p C3-common first star|1.3491|0.48918/0.48861/0.97751|0.98073/0.98071/0.98884|0.125432|
|5p independent cosine amplitudes|1.3763|0.94681/0.94566/0.82714|0.98837/0.98843/0.98611|0.084150|
|7p independent first-star Fourier|1.3623|0.93137/0.93476/0.79652|0.98147/0.97692/0.97668|0.117918|
|3p anisotropic cosine ablation|1.3771|0.95741/0.95625/0.83456|0.98855/0.98860/0.98624|0.082297|

The5p run reached its100-evaluation budget; its final directional-gradient error is2.06e-8. It is not labelled converged. The7p result is a separately counted7-coefficient model. The final anisotropic3p candidate is a **parameter-removal ablation of5p, without another optimization**: set the numerically tiny common sine to zero and impose C1=C3 as indicated independently by the source. Its physical functional has exactly three numbers; no native-fitted coordinate origin is hidden.

Minimal-field candidate for this different target (its tiny K gap remains uncalibrated):

\[
h=-A\nabla^2+2C_a[\cos(G_1\cdot r)+\cos(G_3\cdot r)]
+2C_b\cos(G_2\cdot r),\qquad
(G_1,G_2,G_3)=(b_1,b_2-b_1,-b_2).
\]

The reciprocal entries are
\[
h_{QQ'}(k)=A|k+Q|^2\delta_{QQ'}
+C_a\sum_{g=\pm G_1,\pm G_3}\delta_{Q-Q',g}
+C_b\sum_{g=\pm G_2}\delta_{Q-Q',g}.
\]

It is one spatial scalar field with a spectator physical spin pair. Only a Laplacian and a real bounded periodic potential are needed. A>0 guarantees high-momentum stability. Native individual spin branches remain in all plots, and raw pair probabilities are never replaced by normalized capture or larger-cluster coverage. Grey open circles mark unresolved model multiplets; those individual P values are undefined, not missing calculations.

Coefficients at this angle:

- A = 11.7315129 eV Å²; effective mass = 0.3247648 m_e.
- Ca = 5.80778404 meV.
- Cb = 5.87056607 meV.

The native three-pair energy window is182.6565meV; the final RMS1.3771meV is about.754% of that window. **This does not imply accurate microscopic gaps:** at K the true pair-centre gap is.159059meV and the spin-resolved edge gap is.120921meV, whereas this three-term ablation gives.082297meV. Native first/second-pair spin splittings there are.04460/.03168meV. Global gap-RMS normalization underweights this small gap; a future joint-angle fit must explicitly address the near-K reference gap and declare whether it targets pair centres or actual edges. The present candidate is not labelled fully qualified, and its good pair-projector overlap does not prove SOC is negligible for the tiny edge gap. The lowest conduction pair is~59.7meV wide, so this is much broader than the original valence target. Correlation strength still depends on filling and dielectric screening. No interaction or runtime benchmark was performed.

## Why equal-amplitude3p fails at K

The original3p form A p²+Vc c(r)+Vs s(r) forces the same cosine/sine coefficient on all three first-star directions. Its raw P drops to~.489 for the first two pairs specifically at K, while its mean exceeds.98. A warm start from7p returns to the same solution, excluding the proposed initialization-basin explanation.

We independently computed the exact source Feshbach Schur matrix
\(H_{eff}(E)=EI-[L^\dagger(E-H_\Gamma)^{-1}L]^{-1}\)
in the single-composite space. E is the bare source conduction edge minus50meV, not a fitted model number. The full-source resolvent residual is1.59e-13. First-star source phases give a tiny fixed coordinate shift (~−.000288,−.000290 in moire fractions). C,L and H are transformed consistently; no native wave or7p fitted origin determines this gauge. The correction leaves the equal-amplitude3p K overlap near.489, so coordinate origin is not the main cause here.

The source itself instead has low-Q first-star cosine amplitudes5.6543/5.7610/5.6548meV: a .1067meV difference, only1.875% in norm but comparable to the native first/second-pair gap at K (.1591meV). Ignoring that weak anisotropy imposes an overly strong C3 constraint precisely in a nearly degenerate subspace. Keeping only two distinct real amplitudes is sufficient in this measured ablation; source phases and source C1≈C3 symmetry are independently checked. This is why the successful three-term functional differs from the failed three-term one.

The two lowest pairs combined into a rank4 projector retain~.978 overlap even for the equal-amplitude model at K. That is a separately defined group quantity, not a substitute for either raw individual P. The anisotropic model resolves the individual-state problem without adding orbitals. Near-gap sensitivity makes full-matrix-norm ranking or global bandwidth-normalized RMS alone insufficient tests.

## Independent validation

Fresh six-k diagonalizations reproduce saved energies and raw P to floating-point precision. Real-space/Fourier quadrature and spinless TR were independently verified. The final three-term source-derived anisotropic candidate reaches almost the maximum scalar P available in this fixed source frame for the first two pairs. Model Q37→61 was checked at all60k; it does not constitute source-Q convergence of the underlying TAPW database.

All60 model-Q change: 1.922e-08 meV; Fourier maximum discrepancy: 8.848e-18 eV; fresh raw P discrepancy: 3.331e-16.

All source audits, fit/ablation artifacts and plots are durable in this alternative_targets directory. Main source reconstruction, source-origin solve, fits, parameter removals and fresh verification exited0; source5 optimizer budget stop is retained explicitly. Runtime: gpuh201, /data/home/zy/mambaforge/envs/moirekp (Python3.11.15), at most one4-thread process. Exact source/model code and hashes are in CBM_PILOT_VERIFICATION.json. No old AB valence result, native valence wavefile, or main redo agent's directory was modified.
