from pathlib import Path
import argparse,json,sys,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'source'))
from mcstats import summary
from train import atomic_json
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
records=[json.loads(p.read_text()) for p in (a.base/'pilot/records').glob('pilot_*.json')]
assert len(records)==12
candidates=[]
for r in records:
    with np.load(r['trace']) as t:stats=summary(t['energy'][:,:,0].real)
    se=max(stats['independent_walker_se'],stats['block_se']['128'])
    candidates.append(dict(coupling=r['coupling'],proxy_radius=r['proxy_radius'],
        variance_time=se**2*(r['sampler_seconds']+r['energy_seconds']),
        rhat=stats['rhat']['rank'],statistics=stats,acceptance=r['acceptance']))
valid=[c for c in candidates if c['rhat']<1.03]
assert valid,'All Wolff pilots fail the energy mixing check.'
best=min(valid,key=lambda c:c['variance_time'])
atomic_json(a.base/'frozen/wolff_tuning.json',dict(coupling=best['coupling'],proxy_radius=best['proxy_radius'],
    pilot_only=True,selection_metric='max(replicate SE, batch128 SE)^2 times sampler+energy cost; production independently evaluated',
    candidates=candidates))
print('TUNED',best['coupling'],best['proxy_radius'],flush=True)
