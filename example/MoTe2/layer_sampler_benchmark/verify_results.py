"""Separate successful computation from scientific calibration acceptance."""
from pathlib import Path
import argparse,json,hashlib,numpy as np,sys
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();b=a.base
s=json.loads((b/'production/summary_with_accuracy.json').read_text());failures=[]
sha=hashlib.sha256((b/'frozen/checkpoint.pkl').read_bytes()).hexdigest()
assert sha==s['checkpoint_sha256']
records=list((b/'production/records').glob('*.json'));assert len(records)==44
for path in records:
    r=json.loads(path.read_text());assert r['checkpoint_sha256']==sha,path
    with np.load(r['trace']) as trace:
        assert trace['energy'].shape[0]==r['measurements']
        assert np.isfinite(trace['energy']).all() and np.isfinite(trace['observables']).all(),path
        assert (trace['times']>0).all(),path
for name,g in s['methods'].items():
    failures.extend(f'{name}:{key}' for key,value in g['calibration_checks'].items() if not value)
    if g['time_to_precision']['0.01'] is None:failures.append(name+':0.01meV/hole_SE_not_observed')
    if g['time_to_accuracy_95']['0.01'] is None:failures.append(name+':0.01meV/hole_95pct_accuracy_not_observed')
result=dict(scientific_acceptance_passed=not failures,failures=failures,
    state='certified' if not failures else 'needs_more_sampling_or_investigation',
    expected_cases=44,checkpoint_sha256=sha,
    measurements_per_main_chain={name:max(map(int,g['curves'])) for name,g in s['methods'].items()})
(b/'SCIENTIFIC_STATUS.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
sys.exit(0 if not failures else 3)
