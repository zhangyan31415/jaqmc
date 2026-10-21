# Spin-free M-valley trial: definition and limits

All angles are jointly optimized, independently for SnS2 and SnSe2. Fits reuse the same viewer-Q native eigenvectors projected into the dressed heff basis. No wave-function rotation, renormalization, band reassignment, or fitted per-band energy shift is allowed. This is not a new non-SOC DFT calculation.

## Real-space Hamiltonian

In the source envelope convention, with r in Angstrom and energy in eV,

\[
h=\begin{pmatrix}K_1+U_1(\mathbf r)&T(\mathbf r)\\T(\mathbf r)&K_2+U_2(\mathbf r)\end{pmatrix},\qquad H=h\otimes I_{\rm spin},
\]
\[
K_l=-A_\parallel(\mathbf e_l\cdot\nabla)^2-A_\perp(\mathbf e_l^\perp\cdot\nabla)^2,
\quad A_i=\hbar^2/(2m_i),\quad m_i/m_e=3.80998211615486/A_i.
\]

Nine physical parameters are
\(A_\parallel,A_\perp,w_a,w_b,S_d,V_a,V_d,S_a,S_l\).
The layer axes are \(\mathbf e_l=\boldsymbol\kappa_l/|\boldsymbol\kappa_l|\), as in the parent spinful model. With \(\eta_1=+1,\eta_2=-1\),

\[
U_l=U_e+\eta_l U_o,
\]
\[
U_e=2S_d[\sin(\mathbf G_a\cdot\mathbf r)-\sin(\mathbf G_b\cdot\mathbf r)]+2V_a\cos(\mathbf G_0\cdot\mathbf r)+2S_a\sin(\mathbf G_0\cdot\mathbf r),
\]
\[
U_o=2V_d[\cos(\mathbf G_a\cdot\mathbf r)-\cos(\mathbf G_b\cdot\mathbf r)]+2S_l[\sin(\mathbf G_a\cdot\mathbf r)+\sin(\mathbf G_b\cdot\mathbf r)],
\]
\[
T=2w_a\cos(\mathbf q_a\cdot\mathbf r)+2w_b\cos(\mathbf q_b\cdot\mathbf r),\quad \mathbf q_a=(\mathbf B_1-\mathbf B_2)/2,\quad\mathbf q_b=(\mathbf B_1+\mathbf B_2)/2.
\]

The three intralayer first-star vectors satisfy G_a-G_b=G_0. Their signs/order are fixed by physical.stars(B,kappa1-kappa2,kappa1+kappa2), not chosen independently at each fit. The exact numeric vectors at every angle are in portable/geometry.npz.

The seven-parameter trial imposes V_d=S_l=0 and common axes
\(\mathbf e_1=\mathbf e_2=(\boldsymbol\kappa_1+\boldsymbol\kappa_2)/|\boldsymbol\kappa_1+\boldsymbol\kappa_2|\).
It retains \(A_\parallel,A_\perp,w_a,w_b,S_d,V_a,S_a\). Thus
\[
h_7=(K+U_e)\tau_0+T\tau_x,\qquad h_\pm=K+U_e\pm T.
\]
This implements the zero-twist emergent layer mirror approximation at the level of the local operator. Removing layer-odd potentials alone while keeping the rotated kinetic axes would not enforce it. The original moire translations exchange the +/- parity fields because T changes sign. Scalar parity sectors use the doubled-area cell dual to q_a,q_b, returned by enlarged_cell; no claim is made that one can discard one sector inside the original fixed-k truncated Q basis. No JAQMC solver integration was performed.

## Reciprocal-space matrices

Using plane waves exp[i(k-Q)r], k=k_fractional B,
\[
h_{lQ,lQ'}=\delta_{QQ'}[A_\parallel((k-Q)\cdot e_l)^2+A_\perp((k-Q)\cdot e_l^\perp)^2]+U_l(Q'-Q),
\]
\[
h_{1Q,2Q'}=w_a\sum_{\sigma=\pm1}\delta_{Q-Q',\sigma q_a}+w_b\sum_{\sigma=\pm1}\delta_{Q-Q',\sigma q_b}.
\]
The positive-vector Fourier coefficients of U_l are
\[
U_l(G_a)=-iS_d+\eta_l(V_d-iS_l),\quad
U_l(G_b)=+iS_d+\eta_l(-V_d-iS_l),\quad
U_l(G_0)=V_a-iS_a,
\]
with conjugates at negative vectors, zero otherwise. There are two layer orbitals per valley, and spectator spin degeneracy. Three M valleys are symmetry-related copies; this trial fits the same selected database M valley, not three independently adjusted valleys. No Gamma-valley model is changed here.

## Shared angle dependence and loss

\[
s=\sin^2(\theta/2),\quad x=(s-s_{\min})/(s_{\max}-s_{\min}),\quad u=x/[\rho+(1-\rho)x],
\quad p_j(\theta)=c_{j0}(1-u)^2+2c_{j1}u(1-u)+c_{j2}u^2.
\]
There are 21/27 fitted global angle controls for 7/9 physical coefficients. Fixed inherited rho=0.16056376514075527 for SnS2 and 0.18945221891436223 for SnSe2 is listed separately and is not optimized. Plots draw these exact curves. Off-grid geometry/band accuracy is not certified by drawing the curves.

Native targets are spin-pair centers \(\bar E_{nk}=(E_{2n-1,k}+E_{2n,k})/2\). Both native and model subtract their lowest pair-center energy at Gamma, once per angle. Define reference-only
\[
S_\theta=\max_{n\le3,k}\bar E_{nk}-\min_{n\le3,k}\bar E_{nk},\quad
g_{ak}=\bar E_{a+1,k}-\bar E_{a,k},\quad G_{a\theta}=\sqrt{\langle g_{ak}^2\rangle_k}.
\]
The minimum numerical gap scale is 0.01 meV; no gap reaches this guard in these data. No individual flat-band width divides the loss.

For native group [l,r) among the first six states, C is the fixed, unnormalized native projection and the model orbital spectral projector has rank (r-l)/2:
\[
A^{(lr)}_{QQ'}=\tfrac12\sum_{a=l}^{r-1}\sum_s C^*_{aQs}C_{aQ's},\qquad
P_{lr}=\mathrm{Tr}(A^{(lr)}P^{\rm orbital}_{lr})/[(r-l)/2].
\]
Native exact degeneracies are grouped before evaluation. Minimum principal probabilities are also reported; no trainable spin rotation is used. All60 independent path points are included; the closing Gamma point is displayed from the same packet, without double training weight.

\[
\mathcal L=\left\langle
\left\langle(\delta\bar E/(0.01S_\theta))^2\right\rangle_{3\,bands,k}
+\left\langle(\delta g_a/(0.05G_{a\theta}))^2\right\rangle_{2\,gaps,k}
+\lambda_P\left\langle\{\max(0.96-P_{lr},0)/0.04\}^2\right\rangle_{groups,k}
\right\rangle_\theta
+3\|D(y-y_{seed})\|^2+0.03\|y-y_{seed}\|^2,
\]
where y_j=c_j/scale_j, D=[1,-2,1] for each parameter, and scales in 9p order are [10,10,.1,.05,.02,.02,.02,.02,.02]. Regularization acts on the update from the previous smooth curve after restricting its parameter set. lambda_P=1 for the primary comparison, 10 for additional 9p controls. No extra physical terms are introduced by these weights.

Real spinful edge gaps E3-E2 and E5-E4, energy errors of all six raw states, and absolute meV metrics are reported separately. Fitting pair centers cannot reproduce the native spin splitting. Matching the plotted three centers is therefore weaker than reproducing six spinful bands.

## Spin-free ceiling diagnostic

For one isolated native spin pair, any normalized orbital u tensor a complete two-dimensional spectator spin space satisfies
\[
P=\langle u|A|u\rangle\le\lambda_{\max}(A).
\]
This is a sharp upper bound in the fixed source-derived Kramers representation, including projection-weight loss. It is not an assertion that the projected effective spin is exactly microscopic laboratory spin. Values and all60-point maps are in *_SPINFREE_CEILING.json. Independent pair bounds need not be jointly attainable with mutually orthogonal orbital eigenstates or a seven-/nine-parameter local Hamiltonian.

Across all angles/path points, first-pair bounds are >=0.99955 (SnS2), >=0.99198 (SnSe2); third-pair minima are 0.74973 and 0.76617. At SnS2 9.43° Gamma, the fitted 9p third pair has P=0.24827, with the next model pair carrying P=0.73443: there is both a native spin-mixing ceiling and an incorrect local orbital branch relative to the native pair. Multiplying the overlap weight by ten did not resolve this branch. At SnSe2 9.43° k-index45, P=0.66901 but the same-point ceiling is 0.99864; that failure cannot be attributed to the spin-free ceiling alone. There is no universal claim that all low overlaps are SOC or convention artifacts.

## Literature context

[Calugaru et al., Nature (2025)](https://www.nature.com/articles/s41586-025-09187-5), [preprint and SI](https://arxiv.org/abs/2411.18684v1), discuss emergent layer mirror and approximate effective spin SU(2), including the zero-twist limit and parity-sector translations. [Lei, Mahon and MacDonald, PRL135,196402 (2025)](https://journals.aps.org/prl/abstract/10.1103/5zt2-scbg) and [preprint](https://arxiv.org/html/2411.18828v2) give economical M-valley models. Their simplified-model validity does not establish three-band, all-angle accuracy for the present database structures. Our symmetry/projector diagnostics sample a path, not a full2D BZ.

## Verification and acceptance

Unit tests compare reduced probabilities with explicit spinful Kronecker states, their derivatives with finite differences, native group basis invariance, and SOC-zero parent Hamiltonians. Every fit checks the complete joint residual Jacobian against two consecutive finite-difference steps. Portable exports are checked at all twelve angles and all61 path rows against spectra and dense native group/principal overlaps; enlarged-cell potential periodicity is checked. Expanding model Q by one reciprocal star is a sensitivity test only, not a converged native-Q certificate.

The stated gate requires every angle energy/window<=2%, each pair-center gap RMS/native gap RMS<=10%, group P>=.9, principal>=.8, no unresolved model group boundary, and sampled expanded-Q change<=1meV. All candidates FAIL this gate. A successful optimizer exit is not physical-model validation. The SnSe2 overlap-weight-x10 diagnostic reached its180-evaluation budget and is explicitly not optimizer-converged; it is retained as a finite-budget control, not a stationary result.
