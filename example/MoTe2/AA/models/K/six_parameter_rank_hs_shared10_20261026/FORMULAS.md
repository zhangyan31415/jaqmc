# Hamiltonian and conventions

The stored matrices are the electron valence-band Hamiltonian, in eV. Coordinates r are angstrom, wavevectors inverse angstrom; p=-i grad. The basis is layer-major exp[i(k-Q_l).r]. This is the full envelope gauge, not the periodic layer gauge. q1/q2 and their stored kappas must be retained together. A momentum cutoff makes a Galerkin approximation of the following differential operator.

Let b1,b2 be rows of reciprocal, g=(b1,b2-b1,-b2), h=(b1+b2,b2-2b1,b1-2b2), delta=kappa1-kappa2 and t=(delta,delta+b2,delta+b2-b1). Let s1=+1,s2=-1. Then

    H_e(r,p) = [[-a_h p^2+V_1(r), T(r,p)],
                [T(r,p)^dagger, -a_h p^2+V_2(r)]],
    V_l(r) = 2 sum_j[Vcos cos(g_j.r)-s_l Vsin sin(g_j.r)]
             +2 V2 sum_j cos(h_j.r),
    T0(r) = -w1 sum_j exp(-i t_j.r)-w2 sum_j exp(+2i t_j.r).

Here V_1(r),V_2(r) on the diagonal label layers, whereas the scalar coefficient `V2` labels the second Fourier shell. `a_h=3.80998212/(m_h/m_e)` has units eV angstrom^2; `Vcos,Vsin,w1,V2,w2` have units eV. This negative electron kinetic energy is appropriate to the valence edge; do not use it as an unbounded-below hole kinetic energy. The local potentials are bounded.

For WSe2 only, retain the seventh coefficient lambda (eV angstrom):

    T(r,p)=T0(r)+W(r,p),
    W=-lambda exp(i theta/2) sum_(j=0..2)
         [c_j p_+ exp(-i t_j.r)+c_j^* exp(-i t_j.r) p_-],
    c_j=exp(-2 pi i j/3), p_+=p_x+i p_y, p_-=p_x-i p_y.

Operator ordering matters: the first p acts on the exponential and the wavefunction, and the second p acts on the wavefunction. Theta in exp(i theta/2) is radians. Hermitian conjugation defines the reverse layer block. This first-derivative tunneling is already in response7; it must not be discarded when constructing HF matrix elements. MoTe2 has W=0.

An unambiguous reciprocal-space implementation is the supplied model.py. With d=Q_l-Q_lprime, each first-shell diagonal block is

    Vcos sum_j[1(d=g_j)+1(d=-g_j)]
    -i s_l Vsin sum_j[1(d=g_j)-1(d=-g_j)]
    +V2 sum_j[1(d=h_j)+1(d=-h_j)].

The interlayer block is -w1 sum_j 1(Q1-Q2=t_j)-w2 sum_j 1(Q1-Q2=-2t_j). WSe2 additionally has

    -lambda exp(i theta/2) sum_j 1(Q1-Q2=t_j)
      [c_j ((k-Q1)_x+i(k-Q1)_y)
       +c_j^* ((k-Q2)_x-i(k-Q2)_y)].

The diagonal kinetic term is -a_h |k-Q_l|^2. These formulas fix phases, signs and the derivative ordering without requiring a fit-time gauge. The hole API returns Etop_e(Gamma) I-H_e(-k).T in the conjugate basis. Kprime is H_K(-k).conj(), so K and Kprime are time-reversed selected spin-valley sectors, not an additional free spin index.

Angle laws and exact counts are in FUNCTION.json and MODEL_INFO.json. The HF adapter takes fractional k and uses k_cartesian=k_fractional@reciprocal_basis(theta), with b1,b2 as rows. Archived source band-path fractional coordinates instead use fractional_to_model(theta)=F; source_path_hamiltonian implements that separate convention. To compare a source path point with HF input, use fr_HF=fr_source@F@inverse(B). Neither the local fields nor the single-particle matrices determine the Coulomb form factors or many-body interaction truncation; those must be specified downstream.
