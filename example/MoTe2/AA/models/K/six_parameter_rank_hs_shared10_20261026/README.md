# MoTe2 AA K: six-term10-angle rank_hs/shared_V1_phi_w1

Six physical coefficients: `a_h,Vcos,Vsin,w1,V2,w2`; 10 global fitted coefficients. Shared positive hole mass, negative quadratic electron kinetic energy, first/second-shell local potentials and tunneling. No momentum-dependent tunneling or extra spin variable.

`from hf_adapter import hamiltonian; H=hamiltonian(1.696272693519767,[0,0])` returns the electron matrix in eV. HF fractional k maps with `reciprocal_basis(theta)`. Archived band-path coordinates use the different `fractional_to_model(theta)` and `source_path_hamiltonian` helper. `AngleModel.hamiltonian` takes Cartesian k in inverse angstrom. Exact10calibration geometries and model coefficients are in `cases/`; intermediate geometry is an explicitly labeled interpolation. The angle law is continuous but unvalidated outside the calibration range 1.69627269..5.08584781 degrees.

K is one selected spin-valley sector and Kprime is its time reversal; do not add an independent free spin multiplicity. The optional hole conversion is `Etop(Gamma) I-H_e(-k).T`. Set filling and interactions consistently in the downstream HF code. No HF result or interaction implementation is claimed.

The default rank_hs branch minimizes the selected energy loss, but its new-angle third-state mean overlap squared is only about0.32 and weakest top3 direction about0.31. The wide-seed alternative instead reaches about0.888/0.821/0.824 for the first three states at1.69627 degrees, with top3 RMS0.584meV there and about1.996meV at5.09 degrees; it has higher training loss. Neither is declared a unique qualified material model. The6/10-global-coefficient constraint versions are independently refitted comparisons.

Figures and PDF in `reports` intentionally retain all mode/branch comparisons. Read MODEL_INFO.json/FIT_RESULT.json for the identity of this directory. Archived per-k overlaps compare model with projected Heff, not native TAPW vectors; native energy curves are separate. All10angles trained and no overlap was used in selection. Near degeneracies have separate subspace scores.

See FORMULAS.md for real/reciprocal-space conventions. This is a portable single-particle archive, not a JAQMC training preset or an HF solver.

`CALIBRATION_PARAMETERS.csv` contains only this directory's physical parameters at every calibration angle. Original multi-branch report tables remain under `reports/`.
