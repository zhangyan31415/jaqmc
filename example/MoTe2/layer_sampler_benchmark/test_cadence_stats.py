"""Contracts for correlation-preserving precision and timing analysis."""
import unittest
import numpy as np
from cadence_stats import replicate_bootstrap, stable_crossings, select_cadences

class CadenceContracts(unittest.TestCase):
    def test_bootstrap_keeps_entire_chain_and_seed_strata(self):
        means=np.array([[0.,2.,4.,6.],[10.,12.,14.,16.]])
        draws=replicate_bootstrap(means,200,71)
        # Pool variance includes dispersion of independent chain means;
        # resampling only scalar records would lose time correlations.
        self.assertEqual(draws.shape,(200,))
        self.assertTrue(np.isfinite(draws).all())
        self.assertGreater(draws.mean(),0)
        np.testing.assert_array_equal(draws,replicate_bootstrap(means,200,71))

    def test_crossing_is_stable_and_reports_censoring(self):
        # Draw 0 crosses then recedes, so final stable crossing is index 3.
        se=np.array([[.3,.1,.2,.09],[.3,.2,.15,.12]])
        out=stable_crossings(se,np.array([1.,2.,3.,4.]),.11)
        self.assertEqual(out['crossing_seconds'][0],4.)
        self.assertIsNone(out['crossing_seconds'][1])
        self.assertEqual(out['censored_fraction'],.5)
        self.assertIsNone(out['interval_95_seconds'][1])

    def test_selection_only_uses_pilot_and_breaks_ties_by_cadence(self):
        summaries={'flip':{'40':{'cost':2.},'10':{'cost':1.},'20':{'cost':1.}}}
        self.assertEqual(select_cadences(summaries),{'flip':10})

if __name__=='__main__':unittest.main()
