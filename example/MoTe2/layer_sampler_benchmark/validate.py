"""Physics checks on the actual initialized18-hole projected neural state."""
from pathlib import Path
import json,time,argparse,pickle
from sampler import Sampler,N,coefficients,log_contract,jax,jnp,np
from spectral import initialize,build
from nqs18 import MoireData

parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path)
parser.add_argument('--base',type=Path,default=Path(__file__).resolve().parent)
args=parser.parse_args();base=args.base
data=dict(np.load(base/'bloch_shell5.npz'))
m=Sampler(walkers=2,covariant=True,spectral_g=data['g'],spectral_coeff=data['coeff'])
p,r,s=initialize(m,733,data)
if args.checkpoint:
    with args.checkpoint.open('rb') as f:record=pickle.load(f)
    p=jax.tree_util.tree_map(jnp.asarray,record['params'])
r=r[0];s=s[0]
phi=m.phi(p,r);chi=coefficients(s)
psi=lambda x:jnp.exp(log_contract(m.phi(p,x),chi))
naive=lambda x:jnp.stack(m.wf.apply(p,MoireData(positions=x,spin_coords=s),method=m.wf.naive_projected))*jnp.exp(m.jastrow(x)/N)
np.testing.assert_allclose(phi,naive(r),rtol=1e-9,atol=1e-10)
v=jax.random.normal(jax.random.PRNGKey(71),r.shape)
a=jax.jvp(lambda x:log_contract(m.phi(p,x),chi),(r,),(v,))[1]
b=jax.jvp(lambda x:log_contract(naive(x),chi),(r,),(v,))[1]
np.testing.assert_allclose(a,b,rtol=2e-8,atol=2e-8)
perm=jnp.arange(N).at[0].set(1).at[1].set(0)
original=jnp.exp(log_contract(phi,chi))
exchanged=jnp.exp(log_contract(m.phi(p,r[perm]),chi[perm]))
np.testing.assert_allclose(exchanged,-original,rtol=1e-8,atol=1e-12)
shifted=jnp.exp(log_contract(m.phi(p,r+m.primitive[0]),chi))
np.testing.assert_allclose(shifted,original,rtol=1e-8,atol=1e-12)
energy=jax.jit(m.energy);ec=energy(p,r,chi)
errors=[]
for i in [0,7,17]:
    ch0=chi.at[i].set(jnp.array([1.,0.]));ch1=chi.at[i].set(jnp.array([0.,1.]))
    l0=log_contract(phi,ch0);l1=log_contract(phi,ch1);lc=log_contract(phi,chi)
    ratios=jnp.exp(jnp.array([l0,l1])-lc)*chi[i]
    np.testing.assert_allclose(jnp.sum(ratios),1,rtol=1e-9,atol=1e-9)
    rec=ratios[0]*energy(p,r,ch0)+ratios[1]*energy(p,r,ch1)
    errors.append(float(jnp.max(abs(rec-ec))))
    np.testing.assert_allclose(rec,ec,rtol=2e-6,atol=2e-6)
    angles=jnp.arange(3)*2*jnp.pi/3
    vals=jax.vmap(lambda t:jnp.exp(log_contract(phi,chi.at[i].set(jnp.exp(1j*t*jnp.array([1.,-1.]))))-l0))(angles)
    norm=jnp.mean(abs(vals)**2)
    expected=1+abs(jnp.exp(l1-l0))**2
    np.testing.assert_allclose(norm,expected,rtol=1e-9)
large=build(6)
cutoff=float(np.max(abs(large['eigenvalues']-data['eigenvalues'])))
assert cutoff<1e-3,cutoff
result=dict(projected_reuse_max_error=float(jnp.max(abs(phi-naive(r)))),
 directional_derivative_error=float(abs(a-b)),
 antisymmetry_relative_error=float(abs((exchanged+original)/original)),
 primitive_translation_relative_error=float(abs((shifted-original)/original)),
 local_energy_fourier_errors_meV_per_hole=errors,shell5_vs6_first3bands_max_meV=cutoff)
(base/('validation_trained.json' if args.checkpoint else 'validation.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
