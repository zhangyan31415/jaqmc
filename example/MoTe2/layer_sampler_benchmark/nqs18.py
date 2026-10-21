"""Projected fractional bilayer MoTe2 neural wavefunction and physical energy."""
from pathlib import Path
import os
import sys
import types
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
package=types.ModuleType('jaqmc.app.moire')
package.__path__=[str(ROOT/'src/jaqmc/app/moire')]
sys.modules.setdefault('jaqmc.app.moire',package)
os.environ.setdefault('JAX_ENABLE_X64','true')
import jax
import jax.numpy as jnp
from jaqmc.app.moire.config import MoireConfig
from jaqmc.app.moire.data import MoireData
from jaqmc.app.moire.wavefunction import MoireWavefunction,_combine_translation_and_det_axes
from jaqmc.app.moire.hamiltonian import MoirePotential,MoireSOC
from jaqmc.estimator.kinetic import EuclideanKinetic,LaplacianMode
from jaqmc.estimator.ewald import EwaldSum2D
from jaqmc.wavefunction.input.atomic import SolidFeatures
from jaqmc.utils.supercell import get_reciprocal_vectors
from jaqmc.utils.wiring import runtime_dep
from helpers import microbatch_map

N=18
METHODS=['discrete_flip','conditional','wolff_corrected','continuous']


def coefficients(u,continuous=True):
    return jnp.exp(1j*u[:,None]*jnp.array([1.,-1.])) if continuous else jax.nn.one_hot(u.astype(jnp.int32),2).astype(jnp.complex128)


def log_contract(phi,chi):
    mat=jnp.einsum('il,ldij->dij',chi,phi)
    signs,logs=jnp.linalg.slogdet(mat)
    offset=jnp.max(logs)
    return offset+jnp.log(jnp.sum(signs*jnp.exp(logs-offset)))


class FractionalWavefunction(MoireWavefunction):
    covariant_core: bool=False
    spectral_g: jnp.ndarray | None=runtime_dep(default=None)
    spectral_coeff: jnp.ndarray | None=runtime_dep(default=None)

    def setup(self):
        super().setup()
        if self.covariant_core:
            self.covariant_features=SolidFeatures(simulation_lattice=self.simulation_lattice,
                primitive_lattice=self.primitive_lattice,ae_lattice='primitive',
                distance_type=self.distance_type,sym_type=self.sym_type)

    def _get_orbital_blocks(self,data):
        feature=self.covariant_features if self.covariant_core else self.feature_layer
        embedding=feature(data.positions)
        one=embedding['ae_features'];pair=embedding['ee_features']
        one=one.at[:,0].set(one[:,0]**2)
        pair=pair.at[:,:,0].set(pair[:,:,0]**2)
        hidden,_=self.backbone_layer(one,pair)
        result=[]
        plane=None if self.spectral_g is None else jnp.exp(1j*data.positions@jnp.asarray(self.spectral_g).T)
        for layer,(real,imag) in enumerate(zip(self.real_orbital_layers,self.imag_orbital_layers)):
            phi=jnp.transpose(real(hidden)+1j*imag(hidden),(1,0,2))
            if plane is not None:
                baseline=plane@jnp.asarray(self.spectral_coeff)[layer]
                phi=phi+baseline[None,:,:]
            result.append(phi)
        return tuple(result)

    def _get_momentum_projected_multiphase_orbitals(self,data):
        if not self.covariant_core:
            return super()._get_momentum_projected_multiphase_orbitals(data)
        # Exact reuse for this primitive-periodic core: phi(R-l)=phi(R).
        phi=self._get_orbital_blocks(data)
        translations=jnp.asarray(self.translation_vectors)
        shifted=data.positions[None,:,:]-translations[:,None,:]
        kcom=jnp.asarray(self.k_com)@get_reciprocal_vectors(self.simulation_lattice)
        weights=jnp.exp(1j*(translations@kcom)/sum(self.nspins))
        layer_stack=jnp.stack([weights[:,None,None,None]*p[None,:,:,:] for p in phi],axis=0)
        dressed=self.multiphase_layer(orbital=layer_stack,positions=shifted)
        dressed=self._apply_twist(dressed,shifted)
        return tuple(_combine_translation_and_det_axes(dressed[i]) for i in range(self.nlayers))

    def naive_projected(self,data):
        # Independent slow reference uses all27 explicit neural evaluations.
        return MoireWavefunction._get_momentum_projected_multiphase_orbitals(self,data)


class Model:
    def __init__(self,walkers=32,width=32,depth=3,ndets=1,covariant=False,
                 kinetic_mode='scan',spectral_g=None,spectral_coeff=None):
        self.walkers=walkers
        primitive=np.array([[np.sqrt(3)/2,-.5],[0.,1.]])
        self.cfg=MoireConfig(nelec=N,electron_spins=(N,0),lattice_vectors=primitive.tolist(),
            supercell_matrix=[[3,-3],[3,6]],moire_lattice_constant_nm=.352/(2*np.sin(np.deg2rad(2.)/2)))
        self.cell=jnp.asarray(self.cfg.supercell_lattice)
        self.invcell=jnp.linalg.inv(self.cell)
        self.primitive=jnp.asarray(primitive)
        self.wf=FractionalWavefunction(nspins=(N,0),simulation_lattice=self.cell,
            primitive_lattice=self.primitive,twist=jnp.zeros(2),nlayers=2,
            hidden_dims_single=[width]*depth,hidden_dims_double=[16]*depth,
            ndets=ndets,covariant_core=covariant,spectral_g=spectral_g,spectral_coeff=spectral_coeff)
        self.wf.configure_phase_inputs(self.cfg)
        self.key=jax.random.PRNGKey(20261021)
        self.pref=self.cfg.kinetic_prefactor_mev/self.cfg.effective_mass
        self.kine=EuclideanKinetic(f_log_psi=self.logdata,data_field='positions',
            prefactor=self.pref,mode=LaplacianMode(kinetic_mode))
        self.soc=MoireSOC(f_layer_components=self.components,primitive_lattice=self.primitive,
            nspins=(N,0),prefactor=self.pref)
        self.pot=MoirePotential(f_layer_components=self.components,primitive_lattice=self.primitive,
            nspins=(N,0),v1_mev=self.cfg.v1_mev,phi1_rad=self.cfg.phi1_rad,
            omega1_mev=self.cfg.omega1_mev)
        self.ewald=EwaldSum2D(self.cell)
        self.phi_batch=jax.jit(jax.vmap(self.phi,in_axes=(None,0)))

    def jastrow(self,r):
        i,j=np.triu_indices(N,1)
        theta=2*jnp.pi*((r[i]-r[j])@self.invcell)
        a=jnp.sin(theta)@self.cell/(2*jnp.pi)
        b=(1-jnp.cos(theta))@self.cell/(2*jnp.pi)
        distance=jnp.sqrt(jnp.sum(a*a+b*b,axis=-1)+1e-24)
        slope=self.cfg.coulomb_prefactor_mev/self.pref
        return -jnp.sum(slope*.3/(1+distance/.3))

    def phi(self,p,r):
        phi=self.wf.layer_components(p,MoireData(positions=r,spin_coords=jnp.zeros(N)))[0]
        return phi*jnp.exp(self.jastrow(r)/N)

    def components(self,p,data):
        phi=self.phi(p,data.positions)
        chi=data.spin_coords
        return phi,chi,jnp.einsum('il,ldij->dij',chi,phi)

    def logdata(self,p,data):
        return log_contract(self.phi(p,data.positions),data.spin_coords)

    def energy(self,p,r,chi):
        data=MoireData(positions=r,spin_coords=chi)
        k=self.kine.evaluate_single_walker(p,data,{},None,self.key)[0]['energy:kinetic']
        s=self.soc.evaluate_single_walker(p,data,{},None,self.key)[0]['energy:soc']
        v=self.pot.evaluate_single_walker(p,data,{},None,self.key)[0]['energy:moire_potential']
        c=self.ewald.energy(r,-jnp.ones(N))*self.cfg.coulomb_prefactor_mev
        return jnp.stack([k+s+v+c,k,s,v,c])/N

    def initialize(self,seed=733):
        k1,k2,k3=jax.random.split(jax.random.PRNGKey(seed),3)
        r=jax.random.uniform(k1,(self.walkers,N,2),dtype=jnp.float64)@self.cell
        s=jax.random.uniform(k2,(self.walkers,N),dtype=jnp.float64)*2*jnp.pi
        p=self.wf.init_params(MoireData(positions=r[0],spin_coords=s[0]),k3)
        p=jax.tree_util.tree_map(lambda x:x.astype(jnp.float64),p)
        return p,r,s

    def batch_energy(self,microbatch=1):
        return jax.jit(lambda p,r,chi:microbatch_map(lambda a,b:self.energy(p,a,b),r,chi,microbatch))
