"""Frozen18-hole MoTe2 benchmark. Same checkpoint and energy estimator for all kernels."""
from helpers import task_rank
from pathlib import Path
import argparse,hashlib,json,os,pickle,socket,sys,time
from sampler import Sampler,N,coefficients,jax,jnp,np
from nqs18 import METHODS,ROOT
from observables import make_observables,validate_observable
from train import atomic_json

SEEDS=[30001+37*i for i in range(8)]

def run_case(ex, p, bank, method, seed, count, cadence, kind, obs_batch, output):
    folder=Path(__file__).resolve().parent
    source_files=[folder/name for name in ['benchmark_worker.py','sampler.py','nqs18.py','helpers.py','observables.py']]
    source_files+=sorted((ROOT/'src/jaqmc').rglob('*.py'))
    digest=hashlib.sha256()
    for source in source_files:
        digest.update(str(source.relative_to(ROOT)).encode());digest.update(source.read_bytes())
    source_sha=digest.hexdigest()
    stem=f'{kind}_{method}_{seed}'
    path=output/'traces'/f'{stem}_{count}.npz'
    record_path=output/'records'/f'{stem}.json'
    previous=None;completed=0
    if record_path.exists():
        previous=json.loads(record_path.read_text())
        assert previous['checkpoint_sha256']==ex.checkpoint_sha
        assert previous['sampling_source_sha256']==source_sha
        assert previous['walkers']==ex.walkers
        assert previous['fixed_r_width']==.07
        assert previous['fixed_angle_width']==(np.pi if method=='continuous' else None)
        assert previous['cadence_sweeps']==cadence
        assert previous['coupling']==(getattr(ex,'kernel_coupling',.35) if method=='wolff_corrected' else None)
        assert previous['proxy_radius']==(getattr(ex,'kernel_radius',.65) if method=='wolff_corrected' else None)
        completed=previous['measurements']
        if completed>=count:return previous
    cont = method == 'continuous'
    radius=getattr(ex,'kernel_radius',.65)
    chunk = ex.make_chunk(method, cadence,radius)
    warm_chunk = ex.make_chunk(method, 10,radius)
    state = ex.initial(p, seed, cont)
    family = 'uniform'
    if kind == 'main':
        index = SEEDS.index(seed)
        if index >= 4:
            family = 'trained_bank_perturbed'
            kr = jax.random.fold_in(jax.random.PRNGKey(seed), 727)
            r = jnp.mod((jnp.asarray(bank[:ex.walkers])+.2*jax.random.normal(kr, state[0].shape))@ex.invcell, 1.)@ex.cell
            state = (r, state[1], ex.phi_batch(p, r), state[3])
        if not cont:
            layers = jnp.full_like(state[1], index % 2)
            state = (state[0], layers, state[2], state[3])
    if previous is not None:
        with np.load(previous['trace']) as saved:
            assert len(saved['energy'])==completed
            r=jnp.asarray(saved['final_positions']);u=jnp.asarray(saved['final_internal']);key=jnp.asarray(saved['final_key'])
            phi=jnp.asarray(saved['final_phi_cache'])
        state=(r,u,phi,key)
        family=previous['initial_family']
    coeff_batch = jax.jit(jax.vmap(lambda u: coefficients(u, cont)))
    rw, sw, coupling = .07, np.pi, getattr(ex,'kernel_coupling',.35)
    start = time.perf_counter()
    # Compile using throwaway output; do not change RNG/state for the chain.
    jax.block_until_ready(chunk(p, state, rw, sw, coupling))
    jax.block_until_ready(warm_chunk(p, state, rw, sw, coupling))
    chi = coeff_batch(state[1])
    jax.block_until_ready(ex.energy_batch(p, state[0], chi))
    jax.block_until_ready(obs_batch(state[2], chi, state[0]))
    compile_time = time.perf_counter()-start
    begin_burn = time.perf_counter()
    burn_chunks=2 if kind=='smoke' else 410
    for _ in range(0 if previous else burn_chunks):
        state, _ = warm_chunk(p, state, rw, sw, coupling)
    jax.block_until_ready(state)
    burn_time = time.perf_counter()-begin_burn
    if previous:
        burn_time=previous['burn_seconds']
        compile_time+=previous['compilation_seconds']
    energy = np.empty((count, ex.walkers, 5), complex)
    observables = np.empty((count, ex.walkers, 26), float)
    accepts = np.empty((count, 3), float)
    times = np.empty((count, 4), float)
    if previous:
        with np.load(previous['trace']) as old:
            energy[:completed]=old['energy'];observables[:completed]=old['observables']
            accepts[:completed]=old['acceptance'];times[:completed]=old['times']
    begin = time.perf_counter()
    excluded_io = 0.
    for t in range(completed,count):
        begin_loop = time.perf_counter()
        start = time.perf_counter()
        state, acc = chunk(p, state, rw, sw, coupling)
        jax.block_until_ready(state)
        sampler_time = time.perf_counter()-start
        start = time.perf_counter()
        chi = coeff_batch(state[1])
        e = np.asarray(ex.energy_batch(p, state[0], chi))
        estimator_time = time.perf_counter()-start
        start = time.perf_counter()
        o = np.asarray(obs_batch(state[2], chi, state[0]))
        observable_time = time.perf_counter()-start
        energy[t], observables[t], accepts[t] = e, o, np.asarray(acc)
        times[t] = sampler_time, estimator_time, observable_time, time.perf_counter()-begin_loop
        if not (np.isfinite(e).all() and np.isfinite(o).all()):
            raise FloatingPointError((method, seed, t))
        if (t+1) % 512 == 0 or t+1 == count:
            progress = dict(kind=kind, method=method, seed=seed, completed=t+1, total=count,
                energy_per_hole=float(energy[:t+1, :, 0].real.mean()),
                timed_seconds=float(times[:t+1, 3].sum()), rank=task_rank())
            print('PROGRESS', json.dumps(progress), flush=True)
            io_start = time.perf_counter()
            atomic_json(output/'progress'/f'{stem}.json', progress)
            # Keep resumable sampler state distinct from frozen neural weights.
            np.savez_compressed(output/'states'/f'{stem}.npz', positions=np.asarray(state[0]),
                                internal=np.asarray(state[1]), key=np.asarray(state[3]), completed=t+1)
            excluded_io += time.perf_counter()-io_start
    seconds = float(times[:, 3].sum())
    temporary=path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, energy=energy, observables=observables,
        acceptance=accepts, times=times, final_positions=np.asarray(state[0]),
        final_internal=np.asarray(state[1]),final_key=np.asarray(state[3]),final_phi_cache=np.asarray(state[2]))
    temporary.replace(path)
    record = dict(kind=kind, method=method, seed=seed, measurements=count, walkers=ex.walkers,
        checkpoint_sha256=ex.checkpoint_sha,continued_from=completed,
        sampling_source_sha256=source_sha,
        cadence_sweeps=cadence, burn_sweeps=10*burn_chunks, fixed_r_width=rw, fixed_angle_width=sw if cont else None,
        coupling=coupling if method=='wolff_corrected' else None,
        proxy_radius=radius if method=='wolff_corrected' else None, initial_family=family,
        seconds=seconds, sampler_seconds=float(times[:, 0].sum()),
        energy_seconds=float(times[:, 1].sum()), observable_seconds=float(times[:, 2].sum()),
        compilation_seconds=compile_time, burn_seconds=burn_time,
        elapsed_excluding_checkpoints=time.perf_counter()-begin-excluded_io+(previous['elapsed_excluding_checkpoints'] if previous else 0.),
        mean_energy=float(energy[:, :, 0].real.mean()), imaginary_energy=float(energy[:, :, 0].imag.mean()),
        acceptance=accepts.mean(axis=0).tolist(), trace=str(path),
        rank=task_rank(), host=socket.gethostname(),
        gpu=jax.devices()[0].device_kind, visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
        final_state='successful')
    atomic_json(output/'records'/f'{stem}.json', record)
    print('CASE_DONE', json.dumps(record), flush=True)
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--out',type=Path);parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--records',type=int,default=4096)
    parser.add_argument('--reference-records',type=int,default=8192)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    base=args.base;output=args.out or base/'production'
    for sub in ['traces','states','records','progress']:(output/sub).mkdir(parents=True,exist_ok=True)
    checkpoint=args.checkpoint or base/'frozen/checkpoint.pkl'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with checkpoint.open('rb') as f:record=pickle.load(f)
    if not args.smoke:
        selected=json.loads((base/'frozen/selection.json').read_text())
        assert selected['checkpoint_sha256']==digest
    data=dict(np.load(base/'bloch_shell5.npz'))
    assert hashlib.sha256((base/'bloch_shell5.npz').read_bytes()).hexdigest()==record['configuration']['bloch_sha256']
    ex=Sampler(walkers=32,covariant=True,spectral_g=data['g'],spectral_coeff=data['coeff'])
    ex.checkpoint_sha=digest
    ex.energy_batch=ex.batch_energy(8)
    params=jax.tree_util.tree_map(jnp.asarray,record['params'])
    obs_batch,obs_one=make_observables(ex);residual=validate_observable(ex,params,obs_one)
    rank=task_rank()
    metadata=dict(host=socket.gethostname(),cwd=os.getcwd(),environment=sys.prefix,
        python=sys.executable,jax=jax.__version__,device=jax.devices()[0].device_kind,
        visible=os.environ.get('CUDA_VISIBLE_DEVICES'),job_id=os.environ.get('SLURM_JOB_ID'),
        rank=rank,affinity=sorted(os.sched_getaffinity(0)),checkpoint=str(checkpoint),
        checkpoint_sha256=digest,second_moment_validation_error=residual,
        smoke=args.smoke,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
             for p in Path(__file__).resolve().parent.glob('*.py')})
    atomic_json(output/f'metadata_{rank}.json',metadata);print('METADATA',json.dumps(metadata),flush=True)
    bank=record['positions']
    if args.pilot:
        grid=[(c,r) for c in [0.,.15,.35,.7] for r in [.4,.65,1.]]
        for index,(coupling,radius) in enumerate(grid):
            if index%4!=rank:continue
            ex.kernel_coupling=coupling;ex.kernel_radius=radius
            run_case(ex,params,bank,'wolff_corrected',400100+index*101,512,10,'pilot',obs_batch,output)
    elif args.smoke:
        for method in METHODS:run_case(ex,params,bank,method,819,8,1,'smoke',obs_batch,output)
    else:
        tuning=json.loads((base/'frozen/wolff_tuning.json').read_text())
        ex.kernel_coupling=tuning['coupling'];ex.kernel_radius=tuning['proxy_radius']
        assert len(jax.devices())==1
        for i in range(2):
            run_case(ex,params,bank,METHODS[rank],200001+37*i,1024,1,'fine',obs_batch,output)
        for i,seed in enumerate(SEEDS):
            run_case(ex,params,bank,METHODS[(rank-i)%4],seed,args.records,10,'main',obs_batch,output)
        method='conditional' if rank%2==0 else 'continuous'
        run_case(ex,params,bank,method,900001+101*rank,args.reference_records,10,'reference',obs_batch,output)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    metadata.update(final_state='successful',checkpoint_unchanged=True)
    atomic_json(output/f'metadata_{rank}.json',metadata)
    print('WORKER_SUCCESS',rank,flush=True)

if __name__=='__main__':main()
