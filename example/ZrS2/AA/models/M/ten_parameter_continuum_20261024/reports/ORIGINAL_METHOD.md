# ZrS2 AA M1: ten-parameter continuum model

Selected result: `fit_warped_10p_continuum10`, 10 physical coefficients and 26 independent global coefficients. Its model cutoff is enlarged by two reciprocal stars; all 12 angles and all 720 actual native wave packets are jointly fitted. The nine-parameter base is defined below, followed by the selected common-potential extension.

One orbital per layer, independent of the source-anchored Kramers/pseudospin (its identity matrix is implicit; the projected laboratory spin has not been shown to equal a Pauli matrix). This is a coupled two-layer model, not a single layer-parity scalar model.

\[
H=\begin{pmatrix}K_1+U_e+U_o&T\\T&K_2+U_e-U_o\end{pmatrix},\quad
K_l=-A_\parallel(\mathbf e_l\cdot\nabla)^2-A_\perp(\mathbf e_l^\perp\cdot\nabla)^2.
\]

The unit axes e_l are the actual layer valley axes; both A values are shared constants across all angles. In eV/Angstrom units m/m_e=3.80998211615486/A.

\[
U_e=2S_d[\sin(\mathbf G_a\cdot r)-\sin(\mathbf G_b\cdot r)]+2V_a\cos(\mathbf G_0\cdot r)+2S_a\sin(\mathbf G_0\cdot r),
\]
\[
U_o=2V_d[\cos(\mathbf G_a\cdot r)-\cos(\mathbf G_b\cdot r)]+2S_l[\sin(\mathbf G_a\cdot r)+\sin(\mathbf G_b\cdot r)],\quad
T=2w_a\cos(\mathbf q_a\cdot r)+2w_b\cos(\mathbf q_b\cdot r).
\]

G_a-G_b=G_0. q_a=(B_0-B_1)/2, q_b=(B_0+B_1)/2; all B and actual kappa vectors are in each geometry.npz. The signed first reciprocal star is chosen deterministically from the actual layer valley axes by physical.stars. The same convention is used for native source lift and model.

In the retained layer-Q plane-wave basis, p_l=k B-Q_l with Q_l=kappa_l-g_l, and K_l=A_parallel(p_l.e_l)^2+A_perp(p_l.e_perp_l)^2. A Fourier term c exp(i G.r) connects Q and Q' with Q'-Q=G; its Hermitian partner connects -G. Source plane-wave convention is exp[i(kB-Q).r], as implemented in physical.m_geometry. The two real tunneling cosines connect the two shortest interlayer transfer shells.

The nine coefficients are [A_parallel,A_perp,w_a,w_b,S_d,V_a,V_d,S_a,S_l]. Only the last seven vary with angle. For fixed rho=.18, x=[sin²(theta/2)-sin²(theta_min/2)]/[sin²(theta_max/2)-sin²(theta_min/2)], u=x/[rho+(1-rho)x], and p(theta)=c0(1-u)²+2c1u(1-u)+c2u². Thus the joint fit has 2+7*3=23 adjustable global coefficients; rho is not fitted.

The immutable targets are the database viewer C3-symmetrized M1 TAPW bands and source eigenvectors using the exact stored Q list and reconstructed projector. The first three spin pairs are represented by their centers. One common energy zero is subtracted at Gamma per angle; there are no independent per-band shifts.

The per-angle energy scale is S=max(E_native first3 pairs over k)-min(E_native first3 pairs over k), and each gap scale is G_a=sqrt(mean_k gap_a_native²), floored at 0.01 meV only as a numerical guard. The loss is mean_angles[mean_energy((E_model-E_native)/(.01 S))² + mean_gap((g_model-g_native)/(.05 G))² + mean_group(max(.96-P,0)/.04)²] +3||D(control/scale-seed)||²+.03||control/scale-seed||². The curvature D=(1,-2,1). P is the unchanged native projected-group probability, no renormalization. Native true degeneracies are grouped, and the source-wave projection is independent of the fitted model.

Physical SOC is omitted as an approximation, not by asserting exact spin conservation. The resulting spin-pair-center fit must be assessed alongside raw native overlap and original spin branch energies. Model Q sensitivity and original source Q convergence are separate questions.

## Ten-parameter local-potential extension

A source-informed alternative adds one common-layer coefficient:

\[
U_e\rightarrow U_e+2V_c[\cos(\mathbf G_a\cdot r)+\cos(\mathbf G_b\cdot r)].
\]

It is a local multiplication operator in continuous real space. The model then has10physical coefficients, including the same2constant kinetic coefficients and8quadratic coupling laws, for26independent global fit coefficients. This extends the Fourier content without introducing additional orbitals or momentum-dependent tunneling. Final9/10parameter comparisons use the same native source packets, energy/gap scales and frozen source frame.

## Expanded model-cutoff control

The separate `fit_warped_continuum9` calculation retains the identical nine local real-space operators and the two angle-independent inverse masses, but increases each layer's plane-wave list by two successive additions of the six first-star reciprocal vectors. All twelve angles are jointly refitted; no native energies or native wave packets are regenerated or altered.

Let J inject the original finite-Q orbital/spin coordinates into their exactly matching coordinates in the enlarged model basis. The reference map is extended as C_plus=C J†, with zero new columns. J†J=I and C_plus C_plus†=CC†, verified for every packet. The reported projection is the native-group mean squared singular value of the appropriate block of C_plus(U_model tensor I_spin). There is no row normalization, model-dependent alignment, or interpolated overlap. This is a well-defined comparison with the original finite-Q TAPW target embedded into the larger model space. It does **not** establish the overlap with an independently converged larger-Q atomic/TAPW source, because that source has not been computed.

The enlarged-model fit is initialized from the original nine-parameter result. It uses the same loss and parameter bounds, with the native-action regularization seed recomputed in the enlarged basis. Its global coefficients remain 23. A third-star enlargement at all sixty independent path points is evaluated afterward without refitting, with a single lowest-pair Gamma energy alignment; absolute Gamma shifts are recorded separately. Thus the model-cutoff stability test and the unknown native-source cutoff error remain distinct.


## Material-specific verification and limitations

All 720 native points reproduce the stored viewer energies to at most 1.96e-8 meV; the minimum retained six-state capture is 0.999834. Physical-layer audits at 2.28, 3.89 and 9.43 degrees verify pure bare layer labels; source-resolvent dressing adds at most 1.36% opposite-layer probability.

The scalar source overlap is near unity at small angles. At 7.34 degrees, near an SOC avoided crossing, the second and third native spin pairs have a fixed-frame spin-scalar overlap ceiling of about 0.5356; the first pair remains above 0.9997. Low individual-pair P there cannot be removed by adjusting scalar local potentials alone. The ceiling diagnostic is independent of fitting and never replaces the plotted actual native overlap.

The native-Q 9-parameter result converges but changes by about 5 meV when its model basis is enlarged, so it is retained as a cutoff diagnostic. The expanded-Q 9-parameter optimizer stopped near a very sharp model crossing and failed its production final derivative gate; no successful COMPLETE marker is supplied for that candidate. The selected ten-parameter extension passes the initial and final derivative checks and independent full-path energy, overlap and further-Q recomputation.
