"""Independent SR/KFAC training replicas with frozen holdouts and explicit plateau gate."""
from helpers import task_rank
from pathlib import Path
import argparse,json,pickle,os,socket,time,sys,hashlib
from sampler import Sampler,N,coefficients,jax,jnp,np
from spectral import initialize
from nqs18 import MoireData,microbatch_map
from jax.flatten_util import ravel_pytree
from helpers import source_commit
from optimizers import TrainingUpdate
sys.path.insert(0,str(Path(__file__).resolve().parent/'source'))
from mcstats import summary as analyze_trace,split_rhat


def atomic_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def frozen_eval(m,p,bank,seed,out,records=512):
    """Unclipped energies, fresh RNGs, half uniform/half perturbed training bank."""
    state=m.initial(p,seed,True)
    k=jax.random.PRNGKey(seed+172)
    perturbed=jnp.mod((bank+.2*jax.random.normal(k,bank.shape))@m.invcell,1.)@m.cell
    positions=state[0].at[:m.walkers//2].set(perturbed[:m.walkers//2])
    state=(positions,state[1],m.phi_batch(p,positions),state[3])
    chunk=m.make_chunk('continuous',10);energy=m.batch_energy(8)
    for _ in range(205):state,_=chunk(p,state,.07,jnp.pi,.35)
    values=[];charge=[]
    for i in range(records):
        state,_=chunk(p,state,.07,jnp.pi,.35)
        e=np.asarray(energy(p,state[0],jax.vmap(coefficients)(state[1])))
        if not np.all(np.isfinite(e)):raise FloatingPointError(('evaluation',seed,i))
        values.append(e)
        charge.append(np.asarray(jnp.mean(jnp.cos(2*jnp.pi*(state[0]@m.invcell)),axis=1)))
    values=np.asarray(values)
    charge=np.asarray(charge)
    np.savez_compressed(out,energies=values,charge=charge)
    stats=analyze_trace(values[:,:,0].real)
    # Family difference and early/late drift are evaluated with independent walkers.
    def means_se(x):
        a=x.mean(axis=0);return float(a.mean()),float(a.std(ddof=1)/np.sqrt(len(a)))
    first=means_se(values[:records//2,:,0].real);last=means_se(values[records//2:,:,0].real)
    drift=values[records//2:,:,0].real.mean(axis=0)-values[:records//2,:,0].real.mean(axis=0)
    drift_mean=float(drift.mean());drift_se=float(drift.std(ddof=1)/np.sqrt(m.walkers))
    fam0=means_se(values[:,:m.walkers//2,0].real);fam1=means_se(values[:,m.walkers//2:,0].real)
    summary=dict(statistics=stats,records=records,walkers=m.walkers,mixing_mean_tolerance=.002,
        unclipped=True,mean=float(values[:,:,0].real.mean()),
        se=float(values[:,:,0].real.mean(axis=0).std(ddof=1)/np.sqrt(m.walkers)),
        variance=float(values[:,:,0].real.var()),imaginary_mean=float(values[:,:,0].imag.mean()),
        half_means=[first,last],paired_half_drift=[drift_mean,drift_se],
        family_means=[fam0,fam1],trace=str(out),
        charge_rhat=[split_rhat(charge[:,:,i]) for i in range(2)])
    empirical=summary['se'];block=stats['block_se']['128']
    summary['mixing_pass']=bool(abs(drift_mean)<3*drift_se+.002
        and abs(fam0[0]-fam1[0])<3*np.hypot(fam0[1],fam1[1])+.002
        and stats['rhat']['rank']<1.01 and all(x['rank']<1.02 for x in summary['charge_rhat'])
        and .5<block/max(empirical,1e-30)<2.)
    print('HOLDOUT',json.dumps(summary),flush=True)
    return summary


def train(args):
    rank=task_rank();seed=731+rank*103
    out=args.base/'training'/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    data=dict(np.load(args.base/'bloch_shell5.npz'))
    m=Sampler(walkers=128,covariant=True,kinetic_mode=args.kinetic_mode,
        spectral_g=data['g'],spectral_coeff=data['coeff'])
    p,r,s=initialize(m,seed,data);flat,unravel=ravel_pytree(p)
    state=m.initial(p,seed+10000,True)
    chunk=m.make_chunk('continuous',10);energy=m.batch_energy(8)
    provenance=dict(host=socket.gethostname(),cwd=os.getcwd(),python=sys.executable,
        jax=jax.__version__,device=str(jax.devices()),rank=rank,seed=seed,
        affinity=sorted(os.sched_getaffinity(0)),env_prefix=sys.prefix,
        SLURM_JOB_ID=os.environ.get('SLURM_JOB_ID'),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        parameters=int(flat.size),walkers=m.walkers,threads=os.environ.get('OMP_NUM_THREADS'),
        command=sys.argv,source_directory=str(Path(__file__).resolve().parent),
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
            for f in Path(__file__).resolve().parent.glob('*.py')},
        bloch_input=str(args.base/'bloch_shell5.npz'),
        bloch_sha256=hashlib.sha256((args.base/'bloch_shell5.npz').read_bytes()).hexdigest(),
        native_source_commit=source_commit(),optimizer=args.optimizer,kinetic_mode=args.kinetic_mode)
    atomic_json(out/'provenance.json',provenance);print('ENV',json.dumps(provenance),flush=True)

    optimizer_configuration=dict(optimizer=args.optimizer,kinetic_mode=args.kinetic_mode,
        damping=args.damping,inverse_period=args.kfac_inverse_period,ema=args.kfac_ema)
    history=[];holdouts=[];begin=time.perf_counter();rw=.07;record=None
    resume=out/'latest.pkl'
    start=0
    if resume.exists():
        with resume.open('rb') as f:record=pickle.load(f)
        assert record['configuration']['bloch_sha256']==provenance['bloch_sha256']
        assert record['configuration']['theta_deg']==2 and record['configuration']['holes']==N
        old_config=record.get('optimizer_configuration')
        legacy=(old_config is None and args.optimizer=='sr' and args.kinetic_mode=='scan' and args.damping==.05)
        if old_config!=optimizer_configuration and not legacy and not args.reset_optimizer:
            raise ValueError('Optimizer/backend configuration changed; use --reset-optimizer to migrate weights explicitly.')
        p=jax.tree_util.tree_map(jnp.asarray,record['params'])
        state=(jnp.asarray(record['positions']),jnp.asarray(record['angles']),None,jnp.asarray(record['key']))
        start=record['step'];rw=record['rw'];history=record['history'];holdouts=record['holdouts']
        if args.refine and (out/'converged.pkl').exists():
            old_gate=json.loads((out/'gate.json').read_text())
            if old_gate['passed'] and old_gate['threshold']<=args.plateau_threshold and start>=args.minimum_steps and not args.reset_optimizer:
                print('ALREADY_REFINED',seed,flush=True);return
            (out/'converged.pkl').replace(out/f'coarse_converged_{start}.pkl')
    else:
        for i in range(205):
            state,acc=chunk(p,state,rw,jnp.pi,.35)
            if i<100:rw=float(np.clip(rw*np.exp(.05*(float(acc[0])-.45)),.015,.15))
    trainer=TrainingUpdate(m,p,optimizer=args.optimizer,damping=args.damping,
        inverse_period=args.kfac_inverse_period,ema=args.kfac_ema)
    optimizer_state=trainer.init(p,state[0],state[1],jax.random.fold_in(state[3],201))
    if record is not None and not args.reset_optimizer and record.get('optimizer_state') is not None:
        optimizer_state=jax.tree.map(jnp.asarray,record['optimizer_state'])
    if args.reset_optimizer and holdouts:
        atomic_json(out/f'holdouts_before_reset_{start}.json',holdouts);holdouts=[]
    def save(step,path):
        item=dict(params=jax.device_get(p),positions=np.asarray(state[0]),angles=np.asarray(state[1]),
                  key=np.asarray(state[3]),step=step,rw=rw,history=history,holdouts=holdouts,
                  optimizer_configuration=optimizer_configuration,optimizer_state=jax.device_get(optimizer_state),
                  optimization_gate=dict(threshold=args.plateau_threshold,minimum_steps=args.minimum_steps,
                      holdout_records=args.holdout_records,refinement=args.refine,
                      refinement_lr_multiplier=.5 if args.refine else 1.),
                  provenance=provenance,
                  configuration=dict(theta_deg=2,holes=N,moire_cells=27,filling=-2/3,
                     supercell=[[3,-3],[3,6]],FM=True,hidden_single=[32]*3,hidden_pair=[16]*3,
                     base_ndets=1,projected_ndets=27,covariant_core=True,
                     V_meV=m.cfg.v1_mev,w_meV=m.cfg.omega1_mev,phi_deg=m.cfg.phi1_deg,
                     mass=m.cfg.effective_mass,epsilon=m.cfg.dielectric_constant,
                     period_nm=m.cfg.moire_lattice_constant_nm,jastrow_range_moire=.3,
                     bloch_sha256=provenance['bloch_sha256'],k_com=[0,0],twist=[0,0]))
        tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:pickle.dump(item,f,protocol=5)
        tmp.replace(path)
    strict_previous=(bool(holdouts) and holdouts[-1].get('records',0)>=args.holdout_records
        and holdouts[-1].get('mixing_mean_tolerance',1.)<=.002)
    if args.refine and start>0 and args.max_steps>=args.minimum_steps and not strict_previous:
        baseline=frozen_eval(m,p,state[0],seed+start+170000,
            out/f'refine_start_{start}.npz',args.holdout_records)
        baseline.update(step=start,phase='refinement')
        holdouts.append(baseline);atomic_json(out/'holdouts.json',holdouts);save(start,resume)
    for step in range(start,args.max_steps):
        lr=(.02 if step<2000 else .01 if step<4000 else .005 if step<6000
            else .0025 if step<10000 else .00125 if step<14000 else .000625)
        if args.refine:lr*=.5
        state=(state[0],state[1],m.phi_batch(p,state[0]),state[3])
        state,acc=chunk(p,state,rw,jnp.pi,.35)
        p,optimizer_state,metrics=trainer.step(p,optimizer_state,state[0],state[1],lr,jax.random.fold_in(state[3],step))
        a=np.asarray(metrics)
        if not np.all(np.isfinite(a)):raise FloatingPointError((step,a))
        if (step+1)%20==0:
            entry=dict(step=step+1,mean=float(a[0]),std=float(a[1]),imag=float(a[2]),
                delta_norm=float(a[3]),metric=float(a[4]),lr=lr,
                acceptance=np.asarray(acc).tolist(),seconds=time.perf_counter()-begin)
            history.append(entry);print('TRAIN',json.dumps(entry),flush=True)
            atomic_json(out/'history.json',history)
        if (step+1)%200==0:save(step+1,resume)
        if (step+1)%2000==0:
            label=step+1;save(label,out/f'checkpoint_{label}.pkl')
            prefix='refine_holdout' if args.refine else 'holdout'
            ev=frozen_eval(m,p,state[0],seed+label+70000,out/f'{prefix}_{label}.npz',args.holdout_records)
            ev.update(step=label,phase='refinement' if args.refine else 'initial')
            holdouts.append(ev);atomic_json(out/'holdouts.json',holdouts)
            save(label,resume)
            if label>=args.minimum_steps:
                previous=holdouts[-2];ci=abs(ev['mean']-previous['mean'])+1.96*np.hypot(ev['se'],previous['se'])
                # Numerical optimization plateau within this explicit architecture.
                passed=(ci<args.plateau_threshold and ev['mixing_pass'] and previous['mixing_pass']
                        and all(item.get('records',0)>=args.holdout_records
                            and item.get('mixing_mean_tolerance',1.)<=.002 for item in [previous,ev])
                        and .8<ev['variance']/previous['variance']<1.2)
                gate=dict(step=label,plateau_95pct_upper_meV_per_hole=float(ci),
                          threshold=args.plateau_threshold,passed=bool(passed),variance=ev['variance'])
                print('GATE',json.dumps(gate),flush=True);atomic_json(out/'gate.json',gate)
                if passed:
                    save(label,out/'converged.pkl');return
    save(args.max_steps,resume)
    atomic_json(out/'incomplete.json',dict(reason='optimization gate not passed',max_steps=args.max_steps,
        threshold=args.plateau_threshold,refinement=args.refine))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--max-steps',type=int,default=16000)
    parser.add_argument('--minimum-steps',type=int,default=8000)
    parser.add_argument('--plateau-threshold',type=float,default=.02)
    parser.add_argument('--holdout-records',type=int,default=2048)
    parser.add_argument('--refine',action='store_true')
    parser.add_argument('--optimizer',choices=['sr','kfac'],default='sr')
    parser.add_argument('--kinetic-mode',choices=['scan','forward_laplacian'],default='scan')
    parser.add_argument('--damping',type=float,default=.05)
    parser.add_argument('--kfac-inverse-period',type=int,default=10)
    parser.add_argument('--kfac-ema',type=float,default=.95)
    parser.add_argument('--reset-optimizer',action='store_true')
    train(parser.parse_args())
