# InSe / database 7.In2Se2: AB, Gamma conduction

This experiment fits a continuous real-space model with two layer-anchored conduction channels, approximately one spatial orbital per layer. The selected model has ten coefficients; the nine-coefficient first-star control is retained separately. Physical spin is a degenerate spectator in these candidates; SOC is not fitted. The two kinetic coefficients are fitted as constants shared by all twelve angles, not fixed to externally chosen numbers. `CHANNEL_IDENTITY.json` verifies that retained zero-based same-Q levels 76–79 are all above the occupied-state count 76: these channels do not mix retained valence and conduction bands. Their dressed layer polarization is about 0.74–0.92 in the audited cases, so they are not exactly layer-pure orbitals.

## Hamiltonian and coefficients

With the reciprocal vectors stored as the rows of `B`, define the first star

\[
\mathbf g_1=\mathbf B_1,\quad \mathbf g_2=\mathbf B_2-\mathbf B_1,
\quad \mathbf g_3=-\mathbf B_2,
\qquad C(\mathbf r)=2\sum_{j=1}^3\cos(\mathbf g_j\cdot\mathbf r),
\quad S(\mathbf r)=2\sum_{j=1}^3\sin(\mathbf g_j\cdot\mathbf r).
\]

\[
H=\begin{pmatrix}
-A_1\nabla^2+U_e+U_o&T\\T&-A_2\nabla^2+U_e-U_o
\end{pmatrix},\qquad
U_e=V_{e,c}C+V_{e,s}S,\quad U_o=V_{o,c}C+V_{o,s}S,
\quad T=t_0+T_cC+T_sS.
\]

The first-star nine-coefficient control uses `A1,A2,t0,Ve_c,Ve_s,Vo_c,Vo_s,Tc,Ts`. The selected ten-coefficient candidate adds `V2c` through

\[
U_e\longrightarrow U_e+2V_{2c}\sum_{j=1}^3\cos(\mathbf h_j\cdot\mathbf r),\qquad
(\mathbf h_1,\mathbf h_2,\mathbf h_3)=(\mathbf B_1+\mathbf B_2,\mathbf B_2-2\mathbf B_1,\mathbf B_1-2\mathbf B_2).
\]

This is another local scalar potential: it adds neither orbitals nor derivatives. `A1,A2` have units eV Angstrom squared, the other eight coefficients eV. The effective channel masses satisfy \(m_l/m_e=3.80998211615486/A_l\); they should not be identified with independently measured isolated-monolayer masses. The computational plane waves are \(e^{i(\mathbf k+\mathbf Q)\cdot\mathbf r}\), with source-stored Q coordinates. The diagonal kinetic matrix is \(A_l|\mathbf k+\mathbf Q|^2\); each cosine coefficient connects \(\mathbf Q-\mathbf Q'=\pm\mathbf g_j\) (or `±h_j` for `V2c`) with equal amplitudes, while each sine coefficient has amplitudes \(-i\) at `+g` and `+i` at `-g` times its coefficient. `t0` couples identical Q between layers.

The eight coupling amplitudes use quadratic Bernstein polynomials in a common fixed coordinate:

\[
s=\sin^2(\theta/2),\quad x=(s-s_{\min})/(s_{\max}-s_{\min}),
\quad u=x/[0.2+0.8x],\quad
p(\theta)=c_0(1-u)^2+2c_1u(1-u)+c_2u^2.
\]

There are 26 independently fitted global coefficients: two constant masses and eight times three coupling coefficients. The nine-term control has 23. The coordinate warp 0.2 is fixed. This is a two-layer coupled model; no layer-parity decoupling is imposed.

## Targets, loss and overlap

The reference energies are the database viewer's same-Q TAPW conduction energies. Energy targets are spin-pair centers, each dataset with one common Gamma energy origin. The loss uses the native three-pair total energy window and native adjacent pair-center RMS gap scales; an individual flat-band width is never used as a denominator. Energy residuals are divided by 1% of the native window; gap residuals by 5% of their respective native RMS gap, with a 0.01 meV numerical gap floor. Each angle has equal total weight. Wavefunction residuals are `max(0.96-P,0)/0.04`. For angles greater than 3.8903 degrees only the third-pair wave penalty is zero, matching the user's stated priority; all three energy bands, both gaps, the first two wavefunction groups, and every small-angle wavefunction group remain in the objective. All overlaps are still computed and plotted. Curvature and weak-prior penalties on dimensionless coefficient updates have weights 3 and 0.03. The implementation records the full loss definition in `model10_continued/RESULT.json`.

`C=V_native^dagger L` is evaluated directly at all sixty independent path points. At 3.89 degrees and the recovered 2.28 degree source, `L` is the actual per-k, source-anchored dressed Lowdin embedding. For the other angles, it is an adaptive interpolation of source-anchored embeddings followed by positive polar isometrization. This approximation interpolates the embedding, never the native wavefunctions or the final overlaps. Independent exact source embeddings at held-out path points quantify its error. Every case records its frame and validation controls in `reference/CASE/COMPLETE.json`. No eigenvector fitting, native/model gauge alignment, or normalization of `C` is performed.

An exactly degenerate model eigenspace has no unique individual eigenvectors. At such points, the fitting wave projector includes the whole model-degenerate cluster, normalized by the original native rank. The native energy gaps remain separate targets. The output keeps these two quantities distinct:

- `pair_probability`: native pair versus one spectrally resolved model pair; NaN where the model pair is not uniquely defined.
- `model_cluster_coverage`: the native pair's projection into the entire corresponding model-degenerate cluster. This is subspace coverage and must not be labeled a resolved-band overlap.

The model degeneracy tolerance is 1e-9 eV. Arbitrary-basis raw values are retained only for traceability in separately named arrays. No raw-pair claim is made at a model-degenerate boundary. A preliminary source diagnostic already shows strong spin-orbital mixing of the second and third pairs near Gamma at some angles, so excellent lowest-pair overlap does not establish a spin-independent description of all three pairs.

## Execution and verification

All calculations run on user-authorized `gpuh203`, using `/data/home/zy/mambaforge/envs/moirekp` (Python 3.11.15). Missing 2.28 degree native path eigenvectors are regenerated from the same source H/S and exact stored Q list, with explicit native eigen-residual and viewer-energy checks. Four independent k-point workers use eight BLAS threads each; the cached-angle adaptive-source job uses eight threads. Fitting uses four angle workers with one BLAS thread each. Exact commands and completed states are recorded by each worker and fit. A completed process alone is not scientific acceptance; inspect energy, gap, resolved overlap, cluster coverage, embedding controls, and model Q-expansion sensitivity separately.
