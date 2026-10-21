# Projector + direct/indirect gap refinement

The Hamiltonian family is unchanged:8 nominal paper parameters,AB phases fixed to zero,
6 independent real coefficients. The new loss protects top-six energies, the three doublet
projectors, direct gap45 and the sampled indirect gap. It adds no Hamiltonian parameter.

| Model | dense direct gap min / meV | sampled indirect gap / meV | path energy RMS / meV | path min doublet overlap | fresh6 gap RMS / meV | fresh6 min native overlap | fresh6 2D energy RMS / meV |
|---|---:|---:|---:|---:|---:|---:|---:|
| old | 1.1027 | 0.9935 | 1.0139 | 0.8569 | 1.9190 | 0.8039 | 0.6420 |
| direct_indirect_action | 2.6404 | 2.3769 | 1.8947 | 0.9386 | 1.3206 | 0.9474 | 2.2923 |
| projector_gap | 2.3105 | 2.3003 | 1.5614 | 0.9325 | 1.5190 | 0.9392 | 1.8177 |

The projector model was frozen from development data before the final12 native points were generated.
The dense direct-gap minimum and indirect gap are sampled/model diagnostics, not a proof over the full2D BZ.
Increasing gap still costs some overall band accuracy. The old model remains the recommended energy-first baseline;
this model is the gap-preserving alternative. Intermediate direct-only fits are retained as diagnostics and are not recommended.

Artifacts: projector_selected/ contains the editable model; PROJECTOR_VERIFICATION.json has exact metrics;
projector_gap_comparison.png and projector_bands_comparison.png show the tradeoff.
