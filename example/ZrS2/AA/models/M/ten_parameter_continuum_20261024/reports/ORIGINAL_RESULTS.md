# ZrS2 AA M1: completed continuum candidate

Selected model: **10 physical coefficients, 26 global angle-law coefficients, one spatial orbital per layer, both kinetic coefficients exactly angle independent**. All12 angles and720 actual native wave packets were used.

Low-angle (theta<=3.89deg) first-two-pair energy RMS: **0.51484 meV**; minimum native-group overlap: **0.91744**. All-angle three-pair energy RMS:2.20567meV. Relative gap RMS:2.46% /8.38%. Further model-Q enlargement changes the aligned first3 bands by at most0.01499meV.

Plots: [Bands and native overlap](figures_continuum10/ALL_ANGLES_BANDS.png), [lowest band](figures_continuum10/LOWEST_BAND.png), [parameters](figures_continuum10/PARAMETERS.png), [overlap](figures_continuum10/OVERLAP.png), [spin-scalar ceiling diagnostic](OVERLAP_CEILING_10p.png).

The model retains the simple two-layer kinetic-plus-local-potential form. Relative to the9-coefficient base, its only added term is the common-layer potential2Vc[cos(Ga.r)+cos(Gb.r)]. Real/reciprocal formulas, angle law, loss and source conventions are inMETHOD.md. Parameters and global controls are infigures_continuum10/PARAMETERS.csv andGLOBAL_CONTROLS.csv.

At7.34deg the second and third native spin pairs near an SOC crossing have a scalar overlap ceiling near0.5356; the low P there is not solely a fitting/convention issue. The first pair has ceiling>0.9997. Native-source Q convergence has not been established; only the enlarged model cutoff was checked.

The native-Q9-coefficient result is retained as a cutoff diagnostic. The expanded-Q9-coefficient optimizer stalled at a sharp second/third crossing and failed its production derivative gate; it was not promoted. Full evidence and process exits are recorded inDELIVERY.json and the per-model process files.