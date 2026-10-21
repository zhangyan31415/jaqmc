"""Cross-replica gates and fresh holdout before publishing one frozen checkpoint."""
from helpers import task_rank
from pathlib import Path
import argparse,json,pickle,hashlib,shutil,os,sys
from train import frozen_eval,atomic_json
from sampler import Sampler,jax,jnp,np

p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
p.add_argument('--evaluate',action='store_true');args=p.parse_args();base=args.base
candidates=[];replica_status=[]
for directory in sorted((base/'training').glob('seed*')):
    path=directory/'converged.pkl'
    if not path.exists():
        assert (directory/'incomplete.json').exists(),f'Unfinished training replica: {directory}'
        replica_status.append(dict(seed=directory.name,accepted=False,reason='training plateau/mixing gates not passed by step cap'))
        continue
    record=pickle.loads(path.read_bytes());holdouts=record['holdouts'];ev=holdouts[-1];prev=holdouts[-2]
    gate=record.get('optimization_gate',{})
    assert gate.get('threshold',1.)<=.02 and gate.get('refinement') and record['step']>=8000,'Unrefined checkpoint cannot be selected.'
    assert all(item.get('records',0)>=2048 and item.get('mixing_mean_tolerance',1.)<=.002
               and item.get('phase')=='refinement' for item in [prev,ev]),'Two strict refined holdouts required.'
    quantum=[]
    for item in [prev,ev]:
        with np.load(item['trace']) as trace:
            values=trace['energies'][:,:,0]
            quantum.append(float(np.mean(abs(values-values.mean())**2)))
    if not .8<quantum[1]/quantum[0]<1.2:
        replica_status.append(dict(seed=directory.name,accepted=False,reason='complex quantum variance not stable',quantum_variance=quantum))
        continue
    replica_status.append(dict(seed=directory.name,accepted=True,step=record['step']))
    candidates.append(dict(path=str(path),mean=ev['mean'],se=ev['se'],
        step=record['step'],quantum_variance=quantum,holdout=ev))
assert len(replica_status)==4,replica_status
if len(candidates)<2:
    atomic_json(base/'selection_blocked.json',dict(reason='fewer than two replicas passed all plateau/mixing/quantum-variance checks',candidates=candidates))
    raise RuntimeError('No certified common checkpoint yet; retain training artifacts for continuation.')
candidates.sort(key=lambda c:c['mean']);best=candidates[0]
compatible=all(abs(c['mean']-best['mean'])<.06+2*np.hypot(c['se'],best['se']) for c in candidates[1:])
if not compatible:
    atomic_json(base/'selection_blocked.json',dict(reason='independently converged replicas disagree',candidates=candidates))
    raise RuntimeError('Replica energy agreement failed.')
if args.evaluate:
    rank=task_rank()
    if rank>=len(candidates):
        print('NO_CANDIDATE_FOR_RANK',rank,flush=True);sys.exit(0)
    best=candidates[rank]
record=pickle.loads(Path(best['path']).read_bytes())
data=dict(np.load(base/'bloch_shell5.npz'))
blochhash=hashlib.sha256((base/'bloch_shell5.npz').read_bytes()).hexdigest()
assert record['configuration']['bloch_sha256']==blochhash
m=Sampler(walkers=128,covariant=True,spectral_g=data['g'],spectral_coeff=data['coeff'])
params=jax.tree_util.tree_map(jnp.asarray,record['params'])
final=base/'frozen';final.mkdir(exist_ok=True)
candidate_directory=Path(best['path']).parent
holdout_path=candidate_directory/'final_independent_holdout.json'
if args.evaluate:
    holdout=frozen_eval(m,params,jnp.asarray(record['positions']),923109+rank*109,
        candidate_directory/'final_independent_holdout.npz',2048)
    atomic_json(holdout_path,dict(holdout=holdout,
        checkpoint_sha256=hashlib.sha256(Path(best['path']).read_bytes()).hexdigest()))
    print('INDEPENDENT_CANDIDATE_EVALUATION_DONE',rank,flush=True);sys.exit(0)
saved=json.loads(holdout_path.read_text())
assert saved['checkpoint_sha256']==hashlib.sha256(Path(best['path']).read_bytes()).hexdigest()
holdout=saved['holdout']
final_checks=[]
for candidate in candidates:
    item=json.loads((Path(candidate['path']).parent/'final_independent_holdout.json').read_text())
    assert item['checkpoint_sha256']==hashlib.sha256(Path(candidate['path']).read_bytes()).hexdigest()
    ev=item['holdout']
    final_checks.append(dict(path=candidate['path'],mean=ev['mean'],se=ev['se'],mixing_pass=ev['mixing_pass']))
assert sum(c['mixing_pass'] for c in final_checks)>=2,'Fewer than two independent candidate holdouts mixed.'
assert all(abs(c['mean']-holdout['mean'])<.06+2*np.hypot(c['se'],holdout['se'])
           for c in final_checks if c['mixing_pass']),'New candidate holdouts disagree.'
if not holdout['mixing_pass'] or abs(holdout['mean']-best['mean'])>3*np.hypot(holdout['se'],best['se'])+.002:
    atomic_json(base/'selection_blocked.json',dict(reason='new final holdout failed',holdout=holdout,candidates=candidates))
    raise RuntimeError('Independent final holdout failed.')
shutil.copyfile(best['path'],final/'checkpoint.pkl')
sha=hashlib.sha256((final/'checkpoint.pkl').read_bytes()).hexdigest()
selection=dict(candidates=candidates,replica_status=replica_status,chosen=best,independent_final_holdout=holdout,
    cross_replica_final_holdouts=final_checks,
    checkpoint_sha256=sha,bloch_sha256=blochhash,
    convergence_scope='Numerical SR plateau within the stated covariant neural architecture; no claim of exact ground state or FCI topology.')
atomic_json(final/'selection.json',selection)
print('FROZEN',json.dumps(dict(checkpoint=str(final/'checkpoint.pkl'),sha256=sha,mean=holdout['mean'],se=holdout['se'])),flush=True)
