"""Real18-hole fixed-state numerical checks and synchronized backend timings."""
from pathlib import Path
import argparse,hashlib,json,os,pickle,socket,time,importlib.metadata
from helpers import jax,jnp,microbatch_map,source_commit
import numpy as np
from jax.flatten_util import ravel_pytree
from sampler import Sampler
from nqs18 import coefficients,MoireData,ROOT
from optimizers import TrainingUpdate,clip_training_energy


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--bloch',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--repeats',type=int,default=30)
    ap.add_argument('--warmup',type=int,default=20)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    assert args.repeats>=20 and args.repeats%10==0 and args.warmup>=10
    raw=args.checkpoint.read_bytes();record=pickle.loads(raw);data=dict(np.load(args.bloch))
    assert record['configuration']['theta_deg']==2 and record['configuration']['holes']==18
    assert record['configuration']['bloch_sha256']==hashlib.sha256(args.bloch.read_bytes()).hexdigest()
    p=jax.tree.map(jnp.asarray,record['params'])
    r=jnp.asarray(record['positions'][:128]);u=jnp.asarray(record['angles'][:128]);chi=jax.vmap(coefficients)(u)
    assert r.shape==(128,18,2)
    models={mode:Sampler(walkers=128,covariant=True,kinetic_mode=mode,
        spectral_g=data['g'],spectral_coeff=data['coeff']) for mode in ['scan','forward_laplacian']}
    result=dict(host=socket.gethostname(),python=os.sys.executable,jax=jax.__version__,
        devices=[str(d) for d in jax.devices()],source_commit=source_commit(),
        checkpoint_sha256=hashlib.sha256(raw).hexdigest(),bloch_sha256=hashlib.sha256(args.bloch.read_bytes()).hexdigest(),
        fixture='same trained18-hole checkpoint and128 saved walkers; no parameter optimization during timing',
        microbatch=8,repeats=args.repeats,warmup=args.warmup,
        omp_threads=os.environ.get('OMP_NUM_THREADS'),measurements=[],energy={},checks={})
    source_files=sorted((ROOT/'src/jaqmc').rglob('*.py'))+sorted(Path(__file__).parent.glob('*.py'))
    result['source_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}
    distribution=importlib.metadata.distribution('kfac-jax')
    result['kfac_dependency']=json.loads(distribution.read_text('direct_url.json') or '{}')
    energies={}
    for mode,m in models.items():
        fn=m.batch_energy(8);t=time.perf_counter();energies[mode]=fn(p,r,chi);jax.block_until_ready(energies[mode])
        compile_time=time.perf_counter()-t;times=[]
        for _ in range(2):jax.block_until_ready(fn(p,r,chi))
        for _ in range(args.repeats):
            t=time.perf_counter();jax.block_until_ready(fn(p,r,chi));times.append(time.perf_counter()-t)
        result['energy'][mode]=dict(compile_seconds=compile_time,seconds=times,median_seconds=float(np.median(times)))
    np.testing.assert_allclose(energies['scan'],energies['forward_laplacian'],rtol=1e-10,atol=1e-10)
    result['checks']['energy_max_meV_per_hole']=float(jnp.max(abs(energies['scan']-energies['forward_laplacian'])))
    from jaqmc.laplacian import forward_laplacian,make_laplacian_input
    spatial=r[0].reshape(-1)
    spatial_log=lambda v:models['scan'].logdata(p,MoireData(positions=v.reshape(18,2),spin_coords=chi[0]))
    gradient=jax.jit(jax.jacfwd(spatial_log))(spatial)
    laplacian=jax.jit(lambda v:jnp.trace(jax.jacfwd(jax.jacfwd(spatial_log))(v)))(spatial)
    fwd=jax.jit(lambda v:forward_laplacian(spatial_log)(make_laplacian_input(v)))(spatial)
    np.testing.assert_allclose(gradient,fwd.dense_jacobian,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(laplacian,fwd.laplacian,rtol=1e-10,atol=1e-10)
    result['checks']['spatial_gradient_max']=float(jnp.max(abs(gradient-fwd.dense_jacobian)))
    result['checks']['log_laplacian_error']=float(abs(laplacian-fwd.laplacian))
    x,unravel=ravel_pytree(p);m=models['scan']
    def one(a,b):
        def log(v):
            z=m.logdata(unravel(v),MoireData(positions=a,spin_coords=b))
            return jnp.array([z.real,z.imag])
        return jax.jacrev(log)(x)
    jac=jax.jit(lambda:microbatch_map(one,r,chi,8))();jax.block_until_ready(jac)
    score=jac[:,0]+1j*jac[:,1];e=clip_training_energy(energies['scan'][:,0])
    centered=score-score.mean(axis=0);de=e-e.mean()
    force=jnp.real(jnp.conj(centered).T@de)/len(e)
    prototype=TrainingUpdate(m,p)
    reverse,_=ravel_pytree(prototype.force(p,r,u,e));jax.block_until_ready(reverse)
    np.testing.assert_allclose(force,reverse,rtol=1e-8,atol=1e-10)
    result['checks']['jacobian_vs_vjp_force_max']=float(jnp.max(abs(force-reverse)))
    v=jax.random.normal(jax.random.PRNGKey(91),x.shape);v=v/jnp.linalg.norm(v);eps=1e-5
    def loss(vv):return jnp.real(jnp.mean(jnp.conj(prototype.logs(unravel(vv),r,chi))*de))
    numerical=(loss(x+eps*v)-loss(x-eps*v))/(2*eps);analytic=force@v
    np.testing.assert_allclose(numerical,analytic,rtol=1e-5,atol=1e-8)
    result['checks']['force_directional_finite_difference_error']=float(abs(numerical-analytic))
    cases=[]
    for mode,m in models.items():
        for name,period in [('sr',1),('kfac',1),('kfac',10)]:
            label=f'{name}_{mode}_inverse{period}'
            tr=TrainingUpdate(m,p,optimizer=name,inverse_period=period)
            t=time.perf_counter();state=tr.init(p,r,u,jax.random.PRNGKey(18));jax.block_until_ready(state)
            initialization=time.perf_counter()-t
            tags=[]
            if name=='kfac':
                tags=[eq.params['meta'].variant for eq in tr.kfac._preconditioner.estimator.jaxpr.layer_tags]
                assert 'generic' not in tags,tags
            t=time.perf_counter();new,state,metrics=tr.step(p,state,r,u,.01,jax.random.PRNGKey(0));jax.block_until_ready(metrics)
            first=time.perf_counter()-t
            for i in range(args.warmup):
                new,state,metrics=tr.step(p,state,r,u,.01,jax.random.PRNGKey(100+i))
            jax.block_until_ready(metrics)
            # Pickle roundtrip includes EMA, cached inverses and count, not only params.
            restored=jax.tree.map(jnp.asarray,pickle.loads(pickle.dumps(jax.device_get(state),protocol=5)))
            a=tr.step(p,state,r,u,.01,jax.random.PRNGKey(881));b=tr.step(p,restored,r,u,.01,jax.random.PRNGKey(881))
            for aa,bb in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
                np.testing.assert_allclose(aa,bb,rtol=1e-10,atol=1e-10)
            step,_=ravel_pytree(new);descent=float(force@(step-x));assert descent<0,descent
            initial=tr.init(p,r,u,jax.random.PRNGKey(18))
            cases.append(dict(label=label,trainer=tr,state=state,initial_state=initial,times=[],initialization_seconds=initialization,
                              first_step_seconds=first,tags=tags,force_dot_parameter_step=descent))
            print('COMPILED',label,initialization,first,flush=True)
    # Alternate case order to reduce temperature/order effects; cover inverse refreshes.
    for i in range(args.repeats):
        order=cases if i%2==0 else list(reversed(cases))
        for c in order:
            t=time.perf_counter();new,state,metrics=c['trainer'].step(p,c['state'],r,u,.01,jax.random.PRNGKey(500+i))
            jax.block_until_ready((new,state,metrics));c['times'].append(time.perf_counter()-t);c['state']=state
            assert np.isfinite(np.asarray(metrics)).all()
            step,_=ravel_pytree(new)
            assert float(force@(step-x))<0,c['label']
    # Actual parameter-updating loops from the same initial checkpoint.
    result['training_loops']=[]
    for label in ['sr_scan_inverse1','kfac_forward_laplacian_inverse10']:
        c=next(case for case in cases if case['label']==label);tr=c['trainer'];m=tr.model
        chunk=m.make_chunk('continuous',10)
        carry=(r,u,m.phi_batch(p,r),jax.random.PRNGKey(991))
        jax.block_until_ready(chunk(p,carry,.07,jnp.pi,.35))
        params=p;opt_state=c['initial_state'];elapsed=[];means=[]
        for i in range(args.repeats):
            t=time.perf_counter()
            carry=(carry[0],carry[1],m.phi_batch(params,carry[0]),carry[3])
            carry,_=chunk(params,carry,.07,jnp.pi,.35)
            params,opt_state,metrics=tr.step(params,opt_state,carry[0],carry[1],.01,jax.random.fold_in(carry[3],i))
            jax.block_until_ready((params,opt_state,metrics,carry));elapsed.append(time.perf_counter()-t)
            means.append(float(metrics[0]));assert np.isfinite(np.asarray(metrics)).all()
        result['training_loops'].append(dict(label=label,seconds=elapsed,mean_seconds=float(np.mean(elapsed)),
            energies_meV_per_hole=means,scope='short throughput test with updates; not a convergence comparison'))
    # Sampling cost includes the orbital refresh needed after a parameter update.
    m=models['scan'];chunk=m.make_chunk('continuous',10)
    state=(r,u,m.phi_batch(p,r),jax.random.PRNGKey(991));jax.block_until_ready(chunk(p,state,.07,jnp.pi,.35))
    sample_times=[]
    for _ in range(args.repeats):
        t=time.perf_counter();state=(state[0],state[1],m.phi_batch(p,state[0]),state[3])
        state,_=chunk(p,state,.07,jnp.pi,.35);jax.block_until_ready(state)
        sample_times.append(time.perf_counter()-t)
    result['sampling_seconds']=sample_times
    for c in cases:
        c.pop('trainer');c.pop('state');c.pop('initial_state');c['median_seconds']=float(np.median(c['times']))
        c['mean_seconds']=float(np.mean(c['times']))
        c['mean_iteration_with_sampling_seconds']=float(np.mean(c['times'])+np.mean(sample_times))
        result['measurements'].append(c)
    result['checks']['optimizer_state_pickle_roundtrip']=True
    result['checks']['all_updates_are_descent_directions_for_fixed_clipped_force']=True
    result['checks']['all_kfac_parameters_have_dense_tags']=True
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==result['checkpoint_sha256']
    (args.out/'backend_validation.json').write_text(json.dumps(result,indent=2))
    print('VALIDATION_PASS',json.dumps(result['checks']),flush=True)
    for entry in result['measurements']:print(entry['label'],entry['mean_seconds'],flush=True)

if __name__=='__main__':main()
