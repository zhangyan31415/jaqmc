from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
s=json.loads((a.base/'production/summary.json').read_text())
labels={'discrete_flip':'Discrete flip','conditional':'Gibbs','wolff_corrected':'Corrected Wolff','continuous':'Continuous'}
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for name,g in s['methods'].items():
    grid=sorted((int(k),v) for k,v in g['precision_grid'].items())
    t=np.array([r['energy_seconds'] for _,r in grid]);se=np.array([r['se'] for _,r in grid])
    lo=np.array([r['se_range'][0] for _,r in grid]);hi=np.array([r['se_range'][1] for _,r in grid])
    label=labels[name]+(' (uncalibrated)' if not g['calibration_pass'] else '')
    line,=axes[0].loglog(t,se,label=label)
    axes[0].fill_between(t,lo,hi,alpha=.15,color=line.get_color())
    error=abs(np.array([r['mean'] for _,r in grid])-s['combined_reference']['mean'])
    axes[1].loglog(t,np.maximum(error,1e-10),label=label,color=line.get_color())
axes[0].axhline(.01,color='grey',ls=':',lw=1)
axes[1].axhline(s['combined_reference']['se'],color='grey',ls=':',lw=1,label='Reference 1 SE')
axes[0].set_ylabel('Energy standard error (meV / hole)')
axes[1].set_ylabel('|Mean energy - reference| (meV / hole)')
for ax in axes:
    ax.set_xlabel('Measured GPU seconds: sampler + local energy')
    ax.grid(alpha=.2,which='both');ax.legend(fontsize=8)
fig.suptitle(r'MoTe$_2$: 18 holes, 27 cells, $\nu=-2/3$, $\theta=2^\circ$; fixed NQS')
fig.savefig(a.base/'energy_precision_vs_time.png',dpi=200)
print('PLOT',a.base/'energy_precision_vs_time.png',flush=True)
