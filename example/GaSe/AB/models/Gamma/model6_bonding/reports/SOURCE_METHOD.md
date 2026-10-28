# GaSe AB Γ/CBM continuum model

Status: the12-angle source, full10-parameter candidate, scoped single-bonding4/6-parameter candidates, static plots, parameter CSVs and numerical checks are complete. Optimizer convergence does not certify physical accuracy: RESULTS.md reports the remaining angle-dependent errors. The full10p candidate is model10_sourceprior; the one-field comparison is model4_bonding, with model6_bonding as a third-harmonic test.

The source has four conduction states per Q, local band indices 76–79 above occupancy76. At3.89°, the projected physical layer operator has eigenvalues ±0.89608, and projected spin-z eigenvalues ±0.99468/±0.99603. A two-component layer-localized orbital field with a spectator Kramers pair is therefore the initial approximation. The actual native spinful source is retained for every energy and overlap comparison. Small spin splitting alone is not used to claim that SOC is absent. The third native pair has a scalar representability ceiling as low as0.617 near a higher-band crossing, even at3.89°.

## Hamiltonian

With p=−i∇ (Å⁻¹), the9-parameter model is

\[
h_\theta=\begin{pmatrix}
 A_1p^2+V_e(r)+V_o(r)&T(r)\\
 T(r)&A_2p^2+V_e(r)-V_o(r)
\end{pmatrix},\qquad H=h_\theta\otimes I_2.
\]

\[
V_e=V_{ec}c_1+V_{es}s_1,\quad V_o=V_{oc}c_1+V_{os}s_1,\quad T=t_0+T_cc_1+T_ss_1,
\]

\[
c_1(r)=2\sum_{j=1}^3\cos(G_{1j}\cdot r),\quad s_1(r)=2\sum_{j=1}^3\sin(G_{1j}\cdot r),
\quad \{G_{1j}\}=\{B_0,B_1-B_0,-B_1\}.
\]

All coefficients are real. A1,A2>0 are angle-independent constants; effective masses are m_l/m_e=3.80998211615486/A_l. The remaining seven amplitudes vary with angle. The optional10th physical coefficient is a common second-star cosine potential V2c c2 I, with second star {B0+B1,B1−2B0,B0−2B1}. It introduces no new derivatives. The9-parameter baseline must be assessed before deciding whether this extra term is needed.

For plane waves exp[i(k+q)·r], kinetic diagonal entries are A_l|k+q|²; a +G transfer for amplitude Vc c+Vs s has coefficient Vc−iVs, and a −G transfer has its conjugate. t0 connects equal-q layer components. This Fourier convention is independently checked by real-space unit-cell quadrature. The spin factor does not add a fitted SOC parameter.

## Angle law and loss

For all12 catalog angles, x=[sin²(θ/2)−sin²(θ_min/2)]/[sin²(θ_max/2)−sin²(θ_min/2)] and u=x/[0.2+0.8x]. Each nonkinetic amplitude uses V=b0(1−u)²+2b1u(1−u)+b2u². The9-parameter model has23 global coefficients, and the10-parameter model26. The warp0.2 is fixed. Both kinetic coefficients are exactly shared across all angles.

Reference energies are centers of consecutive pairs in the original energy-ordered TAPW spectrum. Away from time-reversal-invariant k, these are spectral target pairs, not asserted exact Kramers doublets; each angle uses a single Γ-lowest-pair energy zero. Energy residuals divide by0.01 times the original three-pair energy window. Adjacent-gap residuals divide by0.05 times each native gap's path RMS (floor0.01meV). No individual flat-band width is used as a denominator. Angle and k sampling weights are equal.

Wavefunction residuals are max(0,0.96 P_ceiling−P)/0.04. P_ceiling is a fixed source scalar-representation bound, not a normalization of P. For the refined10-parameter fit, all training wave projectors use the same smooth fixed-rank Fermi occupation (0.01meV), including Γ; there is no model-rank-dependent branch switch. At Γ, a native single-pair density uses the fixed symmetry weight

\[
P_A=(I+R+R^2)/3,\quad P_E=I-P_A,\quad
\rho_{TR}=(\rho+T\rho^*T^T)/2,\quad
\widetilde\rho=(P_A+\sqrt2P_E)\rho_{TR}(P_A+\sqrt2P_E).
\]

Here R is the actual-Q120° permutation, and T exchanges q and−q for spinless time reversal. This factor2 restores full conjugate-E coverage when a rank1 Fermi projector half-occupies each member of a protected orbital doublet, and remains continuous as an A level passes that doublet. The fixed source bound is max[λ_max(P_A ρ_TR P_A),2λ_max(P_ω ρ_TR P_ω)], explicitly respecting the spinless-TR paired E constraint. Native rank2 groups and all non-Γ points retain the ordinary density. This is a training symmetry-cluster surrogate, not a rescaled displayed probability. Original native energies/gaps remain separate. The earlier rank-triggeredΓ branch is retained only in the9p diagnostic and the failed10p archive; the latter failed final Jacobian at an A/E exchange and is not a completed model. Final measured native probabilities are unsmeared and are never divided by P_ceiling or source capture. For θ>3.8903°, only the third-pair wave penalty is omitted; all three energy bands, both gaps and the first two pair wavefunctions remain in training at every angle. Curvature/prior penalties are respectively3 and0.03 on scaled-control updates. The selected10p center is SOURCE_PHYSICAL_PRIOR.json; the one-field centers are its fixed-bonding projection plus measured source third-star coefficients when used. The source-action center belongs only to the archived9p diagnostic.

## Native states, source embedding and overlaps

Use the exact stored viewer Q arrays, ordering, raw SOC H/S route and original band files recorded in SELECTION.json. At3.89°, all60 native vectors and all60 source Lowdin lifts are reused from the certified calculation. Other angles with cached native waves use adaptive source-anchor lifts and positive-polar interpolation; every C(k)=V_native(k)†L(k) is still computed from the actual native wavefunction at that k. Source frames, not overlaps, are interpolated. Held-out true lifts are checked at10/30/50 unless already used as an anchor. These embedding approximations are explicitly recorded per angle.

The missing2.28° native vectors are reconstructed from the exact viewer-Q source H, with a shift-invert eigensolver and explicit energy, eigen-residual and Gram checks. One existing exact dense Γ packet is reused. Native degeneracy-block whitening fixes arbitrary nonorthogonal eigenvectors only inside energy-degenerate groups; it is not a model-fit gauge. There is no C renormalization or fitted source/native rotation.

Final overlap is P=Tr(P_native P_model)/rank(P_native), with degenerate native groups treated jointly. An individual model boundary that cuts an exact multiplet has no invariant pair P: that entry is excluded from raw minima, and its full model-cluster coverage is plotted separately with black-edged squares. These two measurements must not be merged in a single claimed pair-P minimum.

## Verification and provenance

All calculations use /data/home/zy/mambaforge/envs/moirekp, Python3.11.15. gpuh203 hosts the main source workers and fits; a released8-thread ZrS2 worker on gpuh201 handled one disjoint2.28deg source segment, and gpuh201 also performed final model verification. The source phase used up to three workers with8 BLAS threads each. Fits started only after12 reference packets were complete, using4 independent angle workers and BLAS1. All outputs remain under this material directory. Baseline source commit e857d8bf9ba04e3dae8fdf093da7b34642e54469; private new scripts and their hashes must accompany the commit.

The final driver writes COMPLETE only after its final Jacobian check. Numerical source validation, initial/final gradient checks, model-Q expansion on all60 independent k points, full native-overlap plots and parameter CSVs are required before a result is called complete. Expanded model-Q sensitivity is not a native TAPW-Q convergence certificate. No JAQMC integration or interacting many-body calculation has been performed.


## Source-certified Gamma ordering

A native Gamma pair is assigned to an orbital A or E sector only when its fixed-source density carries at least98% of its represented weight in that sector. The model Gamma Hamiltonian is diagonalized inside the fixed C3 projectors P_A and P_omega, so A and E energies remain differentiable as their global spectral ranking changes. Source-pure first-three targets impose their corresponding signed order inequalities; the positive separation margin is min(1meV,10% of the adjacent source separation). The fourth native pair provides a reference separation for excluding an A level from the first-three E target window; its full dispersion is not added as a fitted target. Mixed-sector source targets do not impose an unsupported label.

The inequalities add no Hamiltonian coefficient. Their analytic Jacobian was checked independently. The full data loss still includes all12 angles and all three original pair-center energies and two gaps. This avoids a zero-gradient trap when the wrong-symmetry model state has nearly the correct energy but vanishing overlap.

## Physical regularization center

The selected10p version uses source-Gamma q0 tunneling and fixed, low-q Gaussian-weighted source Fourier coefficients as its regularization center. The initial mass center comes from the smallest-angle bonding-block first-shell secant; it is a seed, not a claimed differential monolayer mass. All model coefficients remain freely optimized under their shared angle laws and mass-constant constraints. A source matrix is never inserted into the Hamiltonian as one hidden fitted parameter.

The original native-action initializer had an ill-conditioned tunneling component because a uniform shift of the occupied bonding branch can be removed by the energy-zero convention. Its strongly curved t0 controls should not be treated as a physical prior. SOURCE_PHYSICAL_PRIOR.json records the independently extracted replacement; the complete fit source hashes and coefficient CSVs are retained.

## Single-bonding4/6-parameter alternatives

The stored Gamma Q arrays of both layers are bitwise identical at every angle (BONDING_Q_IDENTITY.json). Therefore the fixed compression C_+=(C_1+C_2)/sqrt2 is well defined. It is never renormalized. One spatial envelope describes the entire bilayer bonding channel, with a spectator spin factor; it is not one spatial orbital per layer.

The4-parameter form is

\[
h_+=-A_b\nabla^2+V_{1c}(\theta)c_1(r)+V_{1s}(\theta)s_1(r)+V_{2c}(\theta)c_2(r).
\]

Its parameters are A_b,V1c,V1s,V2c. A_b is constant across all12 angles and the three potential coefficients each use the same quadratic angle law:10 global coefficients. The6-parameter form adds V3c c3+V3s s3, where G3j=2G1j:16 global coefficients. These are multiplication potentials and the same second-derivative kinetic operator; no higher derivatives or extra orbitals are introduced. The real/reciprocal convention for every star is V(+G)=Vc−iVs and V(−G)=Vc+iVs. The second-star sine is negligible in the audited source, while the third-star amplitudes reach−8.19/+5.22meV at2.28deg, motivating the6p comparison.

Both alternatives are independently jointly refitted, rather than being advertised as exact reductions of the fitted10p Hamiltonian. All12 lowest pairs enter training. Before fitting, a higher target pair is accepted only if its fixed-bonding principal capture is at least0.98 at every path point; this accepts all three pairs through4.41deg and excludes pairs2/3 at5.09deg and above. Their full diagnostic bands and raw overlaps remain in the output and are marked as untrained/outside-sector. Per-angle energy/wave band weights[1,1,0.5] are normalized over eligible pairs; adjacent-gap terms are used only when both pairs are eligible. The original reference three-pair window still sets the energy scale, never an individual flat-band width. The same fixed Gamma symmetry weighting and source-certified ordering apply inside the eligible sector.

High overlap does not imply correct dispersion: the4p model retains P1>=0.980 across all12 angles, but has a3.11meV lowest-band RMS error already at2.28deg and58.98meV at9.43deg. Adding the third star improves small-angle first-two RMS only from5.736 to5.592meV. These comparisons measure the accuracy tradeoff of the smaller NQS field; they do not establish a uniformly accurate full-angle model.

## Final artifact verification

The selected10p,4p and6p versions converged and passed initial/final residual-Jacobian checks and signed-order constraint derivatives. Independent real-space Fourier quadrature checks were performed for the local9/10p and scalar4/6p forms. Actual catalog Q permutations preserve C3 and spinless time reversal to numerical precision. Every saved fitted spectrum was freshly recomputed over720 independent angle/k points. Expanded-Q spectra and native overlaps use projection onto the original native Q rows, with no renormalization; the original database source cutoff itself is not re-converged. Gamma order is retained in the expanded model bases.

Each selected directory has VERIFICATION.json and DIAGNOSTICS.json, including33 held-out true-lift controls. All predictions, band colors and quoted P values are unweighted raw native pair/group probabilities; the fixed Gamma symmetry weight belongs only to the training surrogate. Full model-degenerate cluster coverage is shown separately with black-bordered squares.
