"""Independent pilot/holdout cadence experiment on one immutable checkpoint."""
from pathlib import Path
import argparse,hashlib,json,os,pickle,socket,sys,time
import benchmark_worker as worker
from sampler import Sampler,coefficients,jax,jnp,np
from nqs18 import METHODS
from observables import make_observables,validate_observable
from train import atomic_json

EXPECTED='6f79f2570485b3202cf40fac72c20b230ca286108e1faed5c0bb63a44f6b2423'
CADENCES=[5,10,20,40]
SEEDS={'pilot':[710001+101*i for i in range(8)],'holdout':[810001+101*i for i in range(12)]}


def load_model(base,mode):
    path=base/'frozen/checkpoint.pkl';digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=EXPECTED:raise ValueError(('checkpoint changed',digest))
    with path.open('rb') as f:record=pickle.load(f)
    data=dict(np.load(base/'bloch_shell5.npz'))
    assert hashlib.sha256((base/'bloch_shell5.npz').read_bytes()).hexdigest()==record['configuration']['bloch_sha256']
    model=Sampler(walkers=32,covariant=True,kinetic_mode=mode,spectral_g=data['g'],spectral_coeff=data['coeff'])
    model.kinetic_mode=mode;model.checkpoint_sha=digest;model.energy_batch=model.batch_energy(8)
    tuning=json.loads((base/'frozen/wolff_tuning.json').read_text())
    model.kernel_coupling=tuning['coupling'];model.kernel_radius=tuning['proxy_radius']
    return model,jax.tree_util.tree_map(jnp.asarray,record['params']),record


def validate(base,out):
    fl,p,_=load_model(base,'forward_laplacian');scan,_,_=load_model(base,'scan')
    result=dict(checkpoint_sha256=EXPECTED,host=socket.gethostname(),environment=sys.prefix,
                device=jax.devices()[0].device_kind,checks=[])
    # Actual saved production endpoints from all four representations.
    for method in METHODS:
        records=sorted((base/'production/records').glob(f'main_{method}_*.json'))
        rmeta=json.loads(records[0].read_text())
        with np.load(rmeta['trace']) as t:
            r=jnp.asarray(t['final_positions']);u=jnp.asarray(t['final_internal'])
        chi=jax.vmap(lambda x:coefficients(x,method=='continuous'))(u)
        start=time.perf_counter();a=np.asarray(scan.energy_batch(p,r,chi));scan_first=time.perf_counter()-start
        start=time.perf_counter();b=np.asarray(fl.energy_batch(p,r,chi));fl_first=time.perf_counter()-start
        assert np.isfinite(a).all() and np.isfinite(b).all()
        np.testing.assert_allclose(a,b,rtol=1e-8,atol=2e-7)
        times={}
        for mode,m in [('scan',scan),('forward_laplacian',fl)]:
            trial=[]
            for _ in range(20):
                start=time.perf_counter();jax.block_until_ready(m.energy_batch(p,r,chi));trial.append(time.perf_counter()-start)
            times[mode]=dict(mean_seconds=float(np.mean(trial)),seconds=trial)
        result['checks'].append(dict(method=method,walkers=len(r),max_complex_energy_error=float(np.max(abs(a-b))),
                                    first_call_seconds=dict(scan=scan_first,forward_laplacian=fl_first),timings=times))
    result['passed']=True;atomic_json(out/'backend_validation.json',result)
    print('BACKEND_VALIDATION_PASS',json.dumps(result),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--phase',choices=['validate','smoke','pilot','holdout'],required=True)
    ap.add_argument('--records',type=int);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.phase=='validate':validate(args.base,args.out);return
    (args.out/args.phase).mkdir(parents=True,exist_ok=True)
    assert json.loads((args.out/'backend_validation.json').read_text())['passed']
    ex,p,record=load_model(args.base,'forward_laplacian')
    obs_batch,obs_one=make_observables(ex);obs_error=validate_observable(ex,p,obs_one)
    rank=worker.task_rank();size=int(os.environ.get('SLURM_NTASKS','1'))
    seeds=SEEDS.get(args.phase,[910001]);worker.SEEDS=seeds
    if args.phase=='holdout':
        selection=json.loads((args.out/'selection.json').read_text())
        assert selection['checkpoint_sha256']==EXPECTED
        pairs=[(m,k) for m in METHODS for k in sorted({10,selection['selected_cadences'][m]})]
    else:pairs=[(m,k) for k in CADENCES for m in METHODS]
    count=args.records or (2048 if args.phase=='pilot' else 4096)
    if args.phase=='smoke':count=8
    cases=[(m,k,seed) for seed in seeds for m,k in pairs]
    np.random.default_rng(98201 if args.phase=='pilot' else 98202).shuffle(cases)
    metadata=dict(host=socket.gethostname(),cwd=os.getcwd(),environment=sys.prefix,python=sys.executable,
                  phase=args.phase,rank=rank,world_size=size,job_id=os.environ.get('SLURM_JOB_ID'),
                  device=jax.devices()[0].device_kind,visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  affinity=sorted(os.sched_getaffinity(0)),checkpoint_sha256=EXPECTED,
                  kinetic_mode='forward_laplacian',observable_validation_error=obs_error,
                  commands=sys.argv,assigned_cases=[])
    atomic_json(args.out/args.phase/f'metadata_{rank}.json',metadata)
    for index,(method,cadence,seed) in enumerate(cases):
        if index%size!=rank:continue
        output=args.out/args.phase/f'k{cadence}'
        for sub in ['traces','states','records','progress']:(output/sub).mkdir(parents=True,exist_ok=True)
        r=worker.run_case(ex,p,record['positions'],method,seed,count,cadence,
                          'smoke' if args.phase=='smoke' else 'main',obs_batch,output)
        metadata['assigned_cases'].append(dict(method=method,cadence=cadence,seed=seed,record=str(output/'records'/f"{r['kind']}_{method}_{seed}.json")))
        atomic_json(args.out/args.phase/f'metadata_{rank}.json',metadata)
    assert hashlib.sha256((args.base/'frozen/checkpoint.pkl').read_bytes()).hexdigest()==EXPECTED
    metadata.update(final_state='successful',checkpoint_unchanged=True)
    atomic_json(args.out/args.phase/f'metadata_{rank}.json',metadata)
    print('CADENCE_WORKER_SUCCESS',args.phase,rank,flush=True)

if __name__=='__main__':main()
