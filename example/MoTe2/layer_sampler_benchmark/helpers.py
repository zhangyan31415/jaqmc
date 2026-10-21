"""Small experiment-local numerical helpers; no external research checkout needed."""
import os
import subprocess
from pathlib import Path
os.environ.setdefault('JAX_ENABLE_X64','true')
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64',True)


def task_rank():
    return int(os.environ.get('NQS_RANK',os.environ.get('SLURM_PROCID','0')))


def source_commit():
    result=subprocess.run(['git','-C',str(Path(__file__).resolve().parents[3]),'rev-parse','HEAD'],
                          capture_output=True,text=True,check=False)
    return result.stdout.strip() if result.returncode==0 else None


def microbatch_map(function,positions,angles,microbatch):
    size=positions.shape[0];chunk=min(int(microbatch),size)
    if chunk<1 or size%chunk:
        raise ValueError('Walker count must be divisible by the microbatch size')
    r=positions.reshape(size//chunk,chunk,*positions.shape[1:])
    s=angles.reshape(size//chunk,chunk,*angles.shape[1:])
    batches=jax.lax.map(lambda pair:jax.vmap(function)(pair[0],pair[1]),(r,s))
    return jax.tree_util.tree_map(lambda x:x.reshape(size,*x.shape[2:]),batches)


def sr_direction(log_derivatives,energies,diagonal_shift=.01):
    """Real-parameter SR including amplitude and phase, via sample-space solve."""
    o=log_derivatives-jnp.mean(log_derivatives,axis=0,keepdims=True)
    e=energies-jnp.mean(energies);scale=jnp.sqrt(energies.size)
    a=jnp.concatenate([o.real,o.imag],axis=0)/scale
    b=jnp.concatenate([e.real,e.imag],axis=0)/scale
    dual=a@a.T+diagonal_shift*jnp.eye(2*energies.size)
    return a.T@jnp.linalg.solve(dual,b)
