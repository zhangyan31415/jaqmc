from pathlib import Path
import argparse,json,time
import numpy as np
from nqs18 import Model,N,coefficients,jax,jnp,MoireData

parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--covariant',action='store_true')
parser.add_argument('--mode',default='scan')
parser.add_argument('--walkers',type=int,default=2)
args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
model=Model(walkers=args.walkers,covariant=args.covariant,kinetic_mode=args.mode)
p,r,s=model.initialize()
chi=jax.vmap(coefficients)(s)
fn=model.batch_energy(1)
start=time.perf_counter();e=fn(p,r,chi);jax.block_until_ready(e)
compile_time=time.perf_counter()-start
times=[]
for _ in range(3):
    start=time.perf_counter();jax.block_until_ready(fn(p,r,chi));times.append(time.perf_counter()-start)
validation={}
if args.covariant:
    data=MoireData(positions=r[0],spin_coords=s[0])
    optimized=model.wf.layer_components(p,data)[0]
    naive=model.wf.apply(p,data,method=model.wf.naive_projected)
    naive=jnp.stack(naive,axis=0)
    np.testing.assert_allclose(optimized,naive,rtol=1e-9,atol=1e-9)
    validation['reuse_max_error']=float(jnp.max(abs(optimized-naive)))
result=dict(covariant=args.covariant,kinetic_mode=args.mode,walkers=args.walkers,
    compile_seconds=compile_time,seconds=times,energy=np.asarray(e).tolist(),
    device=jax.devices()[0].device_kind,projected_determinants=27,validation=validation)
# Complex values are encoded explicitly.
result['energy']=[[[float(z.real),float(z.imag)] for z in row] for row in np.asarray(e)]
(args.out/'result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
