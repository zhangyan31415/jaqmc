"""Synthetic numerical contract only; never substitutes for MoTe2 runs."""
import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from cadence_analyze import inspect_group,atomic_json

class AnalysisPipeline(unittest.TestCase):
    def test_whole_walker_precision_pipeline_serializes_and_reports_crossings(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);rng=np.random.default_rng(77);records=[]
            # Stationary iid fixture: analytic mean SE is 1/sqrt(T*S*W).
            for seed in range(8):
                e=rng.normal(size=(2048,32,5))+1j*rng.normal(size=(2048,32,5));e[:,:,0]-=47.7
                path=out/f'{seed}.npz'
                np.savez(path,energy=e,observables=rng.normal(size=(2048,32,26)),times=np.full((2048,4),.001))
                records.append(dict(seed=seed,trace=str(path),checkpoint_sha256='6f79f2570485b3202cf40fac72c20b230ca286108e1faed5c0bb63a44f6b2423',kinetic_mode='forward_laplacian',final_state='successful',measurements=2048,burn_seconds=1.,compilation_seconds=1.,sampler_seconds=1.,energy_seconds=1.,observable_seconds=1.,initial_family='synthetic',method='discrete_flip',cadence_sweeps=10,acceptance=[.3,.4,1.]))
            result,_=inspect_group(records,dict(mean=-47.7,se=.001),200,curves=True)
            atomic_json(out/'contract.json',result)
            self.assertTrue(json.loads((out/'contract.json').read_text())['passed'])
            expected=1/np.sqrt(2048*8*32)
            self.assertLess(abs(result['se']/expected-1),.15)
            self.assertIsNotNone(result['observed_precision_crossings']['0.002']['observed_hot_gpu_seconds'])
            self.assertAlmostEqual(result['timed_gpu_seconds'],2048*8*.002)
            self.assertAlmostEqual(result['cost'],result['se']**2*result['timed_gpu_seconds'])

if __name__=='__main__':unittest.main()
