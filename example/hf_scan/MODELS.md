# Current continuum model packages

Updated 2026-10-29: 16 materials, 37 model versions, 34 uniform one-body matrix adapters.

Physical parameters per angle and global angle-law coefficients are separate counts. Diagnostic and historical versions remain visible; inclusion is not an accuracy certificate.

| Material | Stacking / valley | Current versions | One-body scope |
|---|---|---|---|
| CdCl2 | AA/Gamma | [CdCl2_AA_Gamma_scalar3_schur](../CdCl2/AA/models/Gamma/scalar3_schur/README.md); [CdCl2_AA_Gamma_scalar5_schur](../CdCl2/AA/models/Gamma/scalar5_schur/README.md); [CdCl2_AA_Gamma_scalar7_schur](../CdCl2/AA/models/Gamma/scalar7_schur/README.md) | diagnostic_or_historical; first_band_candidate |
| Ga2S2 | AB/Gamma | [Ga2S2_gamma10_20261020](../Ga2S2/AB/models/Gamma/gamma10_20261020/README.md); [Ga2S2_gamma9_20261021](../Ga2S2/AB/models/Gamma/gamma9_20261021/README.md) | diagnostic_or_historical |
| GaSe | AB/Gamma | [GaSe_AB_Gamma_model10_sourceprior](../GaSe/AB/models/Gamma/model10_sourceprior/README.md); [GaSe_AB_Gamma_model4_bonding](../GaSe/AB/models/Gamma/model4_bonding/README.md); [GaSe_AB_Gamma_model6_bonding](../GaSe/AB/models/Gamma/model6_bonding/README.md) | diagnostic_or_historical; exploratory_restricted_angles |
| HfS2 | AA/M1 | [HfS2_AA_M_nine_parameter_continuum_20261024](../HfS2/AA/models/M/nine_parameter_continuum_20261024/README.md); [HfS2_AA_M_ten_parameter_continuum_20261024](../HfS2/AA/models/M/ten_parameter_continuum_20261024/README.md) | small_angle_candidate |
| HfSe2 | AA/M1 | [HfSe2_AA_M_nine_parameter_continuum_20261023](../HfSe2/AA/models/M/nine_parameter_continuum_20261023/README.md) | small_angle_candidate |
| InSe | AB/Gamma | [InSe_AB_Gamma_model10_continued](../InSe/AB/models/Gamma/model10_continued/README.md); [InSe_AB_Gamma_model9_low2](../InSe/AB/models/Gamma/model9_low2/README.md) | diagnostic_or_historical; exploratory_small_angle |
| MgBr2 | AB/Gamma | [MgBr2_cbm_scalar3_anisotropic](../MgBr2/AB/models/Gamma/cbm_scalar3_anisotropic/README.md); [MgBr2_kane6_order10](../MgBr2/AB/models/Gamma/kane6_order10/README.md); [MgBr2_kane6_replacement10](../MgBr2/AB/models/Gamma/kane6_replacement10/README.md) | diagnostic_or_historical |
| MgCl2 | AA/Gamma | [MgCl2_gamma10_20261020](../MgCl2/AA/models/Gamma/gamma10_20261020/README.md); [MgCl2_gamma9_20261021](../MgCl2/AA/models/Gamma/gamma9_20261021/README.md) | diagnostic_or_historical |
| MoTe2 | AA/K | [MoTe2_AA_K_six_parameter_joint_20261016](../MoTe2/AA/models/K/six_parameter_joint_20261016/README.md); [six_parameter_rank_hs_10angles_20261026](../MoTe2/AA/models/K/six_parameter_rank_hs_10angles_20261026/README.md); [six_parameter_rank_hs_shared10_20261026](../MoTe2/AA/models/K/six_parameter_rank_hs_shared10_20261026/README.md); [six_parameter_rank_hs_shared6_20261026](../MoTe2/AA/models/K/six_parameter_rank_hs_shared6_20261026/README.md); [six_parameter_rank_hs_wide_seed_20261026](../MoTe2/AA/models/K/six_parameter_rank_hs_wide_seed_20261026/README.md) | diagnostic_or_historical; exploratory_tradeoff |
| PtSe2 | AA/Gamma | [PtSe2_chiral10](../PtSe2/AA/models/Gamma/chiral10/README.md); [PtSe2_orbital10](../PtSe2/AA/models/Gamma/orbital10/README.md) | diagnostic_or_historical |
| SnS2 | AA/M1 | [SnS2_AA_M_nine_parameter_constant_mass_20261021](../SnS2/AA/models/M/nine_parameter_constant_mass_20261021/README.md); [SnS2_AA_M_nine_parameter_parallel_mass_only_20261021](../SnS2/AA/models/M/nine_parameter_parallel_mass_only_20261021/README.md); [SnS2_AA_M_seven_parameter_constant_mass_20261021](../SnS2/AA/models/M/seven_parameter_constant_mass_20261021/README.md) | exploratory_cutoff_check_required |
| SnSe2 | AA/M1 | [SnSe2_AA_M_nine_parameter_constant_mass_20261021](../SnSe2/AA/models/M/nine_parameter_constant_mass_20261021/README.md); [SnSe2_AA_M_nine_parameter_parallel_mass_only_20261021](../SnSe2/AA/models/M/nine_parameter_parallel_mass_only_20261021/README.md); [SnSe2_AA_M_seven_parameter_constant_mass_20261021](../SnSe2/AA/models/M/seven_parameter_constant_mass_20261021/README.md) | exploratory_cutoff_check_required |
| WS2 | AA/K1 | [WS2_spectral_20261014](../WS2/AA/models/K/spectral_9p43_20261014/README.md) | diagnostic_or_historical |
| WSe2 | AA/K | [seven_parameter_joint_17angles_20261028](../WSe2/AA/models/K/seven_parameter_joint_17angles_20261028/README.md); [WSe2_AA_K_seven_parameter_top3_20261018](../WSe2/AA/models/K/seven_parameter_top3_20261018/README.md) | calibrated_candidate_with_wave_limits; diagnostic_or_historical |
| WTe2 | AA/K1 | [WTe2_spectral_20261014](../WTe2/AA/models/K/spectral_3p89_20261014/README.md) | diagnostic_or_historical |
| ZrS2 | AA/M1, AB/Gamma | [ZrS2_AA_M_ten_parameter_continuum_20261024](../ZrS2/AA/models/M/ten_parameter_continuum_20261024/README.md); [ZrS2_AB_Gamma_six_parameter_20261014](../ZrS2/AB/models/Gamma/six_parameter_20261014/README.md) | diagnostic_or_historical; small_angle_candidate |

## Read before a scan

- [Commands, conventions and interaction inputs](README.md).
- [Machine-readable model catalogue](MODEL_CATALOG.json) and [CSV](MODEL_CATALOG.csv).
- The old ZrS2 AB six-coefficient target is Gamma with eight internal states per Q. The new ZrS2 AA ten-coefficient target is M1 with one spatial orbital per layer.
- WSe2 seventeen-angle and MoTe2 ten-angle versions are distinct from the historical October18/16 archives.
- No atomic H/S or native-wavefunction cache is required to load the one-body packages. Density vertices and interactions are separate inputs.
- Every new package contains its source-file hashes and numerical replay checks. See [integration verification](VERIFICATION.json) for fresh-process common-adapter checks.
