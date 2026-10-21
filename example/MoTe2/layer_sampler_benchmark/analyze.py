"""Direct long-run precision calibration on independent frozen-kernel walkers."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.stats import norm
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'source'))
from mcstats import summary, independent_mean_se, seed_anova, stable_precision_crossing

BASE = Path(__file__).resolve().parent
METHODS = ['discrete_flip', 'conditional', 'wolff_corrected', 'continuous']
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, default=BASE/'production')
parser.add_argument('--partial', action='store_true')
args = parser.parse_args()
output = args.out
records = [json.loads(p.read_text()) for p in sorted((output/'records').glob('*.json'))]
metas = [json.loads(p.read_text()) for p in sorted(output.glob('metadata_*.json'))]
counts = {kind:sum(r['kind']==kind for r in records) for kind in ['fine', 'main', 'reference']}
result = dict(counts=counts, checkpoint_sha256=metas[0]['checkpoint_sha256'] if metas else None,
              methods={}, fine={}, reference={}, precision_targets=[.02, .01, .005, .002])


def structure(records):
    walker_obs = []
    for r in records:
        with np.load(r['trace']) as t:
            walker_obs.append(t['observables'].mean(axis=0))
    obs = np.concatenate(walker_obs, axis=0)
    m = obs.mean(axis=0)
    spin_mode = m[6:10]+1j*m[10:14]
    charge_mode = m[18:22]+1j*m[22:26]
    connected_spin = m[2:6]-18*abs(spin_mode)**2
    connected_charge = m[14:18]-18*abs(charge_mode)**2
    np.testing.assert_allclose(m[2], 18*m[1], atol=1e-10)
    rng = np.random.default_rng(1987)
    idx = rng.integers(len(obs), size=(3000, len(obs)))
    boot = obs[idx].mean(axis=1)
    boot_spin = boot[:, 2:6]-18*abs(boot[:, 6:10]+1j*boot[:, 10:14])**2
    return dict(polarization=float(m[0]), polarization_second_moment=float(m[1]),
        raw_spin_structure=m[2:6].tolist(), connected_spin_structure=connected_spin.tolist(),
        connected_charge_structure=connected_charge.tolist(),
        connected_spin_bootstrap_se=boot_spin.std(axis=0, ddof=1).tolist(),
        collective_zero_q_ratio=float(connected_spin[0]/max(1-m[0]**2, 1e-12)),
        q_order=['zero', 'supercell_b1', 'supercell_b2', 'supercell_b1_minus_b2'])


def ensemble(records, prefixes):
    records = sorted(records, key=lambda r:r['seed'])
    energies, obs_m, obs_rho, timing = [], [], [], []
    for r in records:
        with np.load(r['trace']) as t:
            energies.append(t['energy'][:, :, 0])
            obs_m.append(t['observables'][:, :, 0])
            obs_rho.append(t['observables'][:, :, 19])
            timing.append(t['times'])
    length = min(len(x) for x in energies)
    x = np.stack([e[:length].real for e in energies], axis=1)  # time,seed,walker
    im = np.stack([e[:length].imag for e in energies], axis=1)
    rng = np.random.default_rng(9231)
    M = x.shape[1]*x.shape[2]
    bootstrap_idx = rng.integers(M, size=(4000, M))
    curves = {}
    for n in prefixes:
        if n > length:
            continue
        xp = x[:n]
        record = summary(xp)
        means = xp.mean(axis=0).ravel()
        record['replicate_means'] = means.tolist()
        boot_se = np.std(means[bootstrap_idx], axis=1, ddof=1)/np.sqrt(M)
        record['empirical_se_bootstrap_range'] = np.quantile(boot_se, [.025, .975]).tolist()
        record['seed_anova'] = seed_anova(xp) if xp.shape[1] > 1 else None
        record['energy_seconds'] = sum(float(t[:n, :2].sum()) for t in timing)
        record['all_observable_seconds'] = sum(float(t[:n, 3].sum()) for t in timing)
        record['with_burn_seconds'] = record['energy_seconds']+sum(r['burn_seconds'] for r in records)
        record['empirical_cost'] = record['independent_walker_se']**2*record['energy_seconds']
        record['ips_cost'] = record['ac']['se']**2*record['energy_seconds']
        record['imaginary_mean'] = float(im[:n].mean())
        record['walker_count'] = M
        record['empirical_gamma'] = record['independent_walker_se']**2*n*M
        if n >= 512:
            block = 128
            flat = xp.reshape(n,M)
            bm = flat[:n//block*block].reshape(n//block,block,M).mean(axis=1)
            per_walker_gamma = np.var(bm,axis=0,ddof=1)*block
            record['batch128_cost'] = record['block_se']['128']**2*record['energy_seconds']
            record['batch128_per_walker_gamma'] = per_walker_gamma.tolist()
        # Compare the early/late halves of every chain without using reference.
        midpoint = n//2
        drift = xp[midpoint:].mean(axis=0)-xp[:midpoint].mean(axis=0)
        drift_mean, drift_se = float(drift.mean()), float(drift.std(ddof=1)/np.sqrt(M))
        record['half_drift'] = dict(mean=drift_mean, se=drift_se, z=drift_mean/max(drift_se,1e-30))
        curves[str(n)] = record
    full = curves[str(max(map(int, curves)))]
    centered=x-x.mean()
    tails=dict(minimum=float(x.min()),maximum=float(x.max()),
        quantiles=np.quantile(x,[.001,.01,.5,.99,.999]).tolist(),
        largest_squared_deviation_fraction=float(np.max(centered**2)/np.sum(centered**2)))
    # A dense prefix grid for observed time-to-precision. Only inexpensive
    # replicate-mean statistics here; full spectral diagnostics stay dyadic.
    cumulative = np.cumsum(x.reshape(length, M), axis=0)
    grid = {}
    for n in range(64, length+1, 64):
        means = cumulative[n-1]/n
        empirical_se = float(means.std(ddof=1)/np.sqrt(M))
        boot = np.std(means[bootstrap_idx[:2000]], axis=1, ddof=1)/np.sqrt(M)
        grid[str(n)] = dict(mean=float(means.mean()), se=empirical_se,
            se_range=np.quantile(boot,[.025,.975]).tolist(),
            energy_seconds=sum(float(t[:n,:2].sum()) for t in timing))
    family_means = {}
    for family in sorted(set(r['initial_family'] for r in records)):
        chosen = [i for i,r in enumerate(records) if r['initial_family']==family]
        values = x[:, chosen]
        family_means[family] = dict(mean=float(values.mean()), se=independent_mean_se(values))
    group = dict(curves=curves, seeds=[r['seed'] for r in records],
        cadence_sweeps=records[0]['cadence_sweeps'], initial_families=family_means,
        final=full, structure=structure(records),
        precision_grid=grid, burn_seconds=sum(r['burn_seconds'] for r in records),
        tails=tails,
        acceptance=np.mean([r['acceptance'] for r in records],axis=0).tolist(),
        rank_method_counts={str(rank):sum(r['rank']==rank for r in records) for rank in range(4)})
    if len(family_means)==2:
        first,second=family_means.values()
        delta=first['mean']-second['mean']
        se=float(np.hypot(first['se'],second['se']))
        group['initial_family_comparison']=dict(difference=delta,se=se,z=delta/se)
    m = np.stack(obs_m, axis=1)
    rho = np.stack(obs_rho, axis=1)
    group['polarization_diagnostics'] = summary(m)
    group['density_diagnostics'] = summary(rho)
    return group, x.mean(axis=0).ravel(), bootstrap_idx


for method in METHODS:
    fine = [r for r in records if r['kind']=='fine' and r['method']==method]
    if fine:
        group, _, _ = ensemble(fine, [128,256,512,1024])
        result['fine'][method] = group
    main = [r for r in records if r['kind']=='main' and r['method']==method]
    if main:
        group, means, bootstrap_idx = ensemble(main, [128,256,512,1024,2048,4096,8192,16384])
        result['methods'][method] = group

reference_records = [r for r in records if r['kind']=='reference']
reference_means = []
for method in ['conditional','continuous']:
    refs = [r for r in reference_records if r['method']==method]
    if refs:
        group, means, _ = ensemble(refs, [1024,2048,4096,8192,16384])
        result['reference'][method] = group
        reference_means.extend(means)
if len(reference_records)==4:
    ref = np.array(reference_means)
    ref_mean = float(ref.mean())
    ref_se = float(ref.std(ddof=1)/np.sqrt(len(ref)))
    a, b = [result['reference'][m]['final'] for m in ['conditional','continuous']]
    z = (a['mean']-b['mean'])/np.hypot(a['independent_walker_se'],b['independent_walker_se'])
    result['combined_reference'] = dict(mean=ref_mean,se=ref_se,agreement_z=float(z),walkers=len(ref))
    reference_checks=dict(representation_agreement=bool(abs(z)<3))
    for name in ['conditional','continuous']:
        group=result['reference'][name]
        reference_checks[name+'_energy_rhat']=group['final']['rhat']['rank']<1.01
        reference_checks[name+'_polarization_rhat']=group['polarization_diagnostics']['rhat']['rank']<1.01
        reference_checks[name+'_density_rhat']=group['density_diagnostics']['rhat']['rank']<1.01
        reference_checks[name+'_half_drift']=abs(group['final']['half_drift']['z'])<3
    result['reference_calibration_checks']=reference_checks
    for method, group in result['methods'].items():
        for n, r in group['curves'].items():
            r['difference_from_reference'] = r['mean']-ref_mean
            r['difference_se'] = float(np.hypot(r['independent_walker_se'],ref_se))
            r['reference_agreement_z'] = r['difference_from_reference']/r['difference_se']
            means=np.array(r['replicate_means'])
            r['reference_corrected_walker_rmse']=float(np.sqrt(max(0.,np.mean((means-ref_mean)**2)-ref_se**2)))
            predicted=r['ac']['se']*np.sqrt(r['walker_count'])
            sigma=np.hypot(predicted,ref_se)
            r['coverage_ips']={str(k):float(np.mean(abs(means-ref_mean)<=k*sigma)) for k in [1.,1.96]}
            if int(n)>=512:
                predicted=r['block_se']['128']*np.sqrt(r['walker_count'])
                sigma=np.hypot(predicted,ref_se)
                r['coverage_batch128']={str(k):float(np.mean(abs(means-ref_mean)<=k*sigma)) for k in [1.,1.96]}
                r['variance_ratio_empirical_to_batch128']=(r['independent_walker_se']/r['block_se']['128'])**2
            r['variance_ratio_empirical_to_ips']=(r['independent_walker_se']/r['ac']['se'])**2
        # Retrospective crossing, requiring all subsequent measured prefixes
        # to pass a conservative SE upper bound. Do not extrapolate thresholds.
        crossings = {}
        for delta in result['precision_targets']:
            crossings[str(delta)] = stable_precision_crossing(group['precision_grid'],delta,group['burn_seconds'])
        group['time_to_precision'] = crossings
        final=group['final']
        group['calibration_checks']=dict(
            energy_rhat=final['rhat']['rank']<1.01,
            polarization_rhat=group['polarization_diagnostics']['rhat']['rank']<1.01,
            density_rhat=group['density_diagnostics']['rhat']['rank']<1.01,
            reference_agreement=abs(final['reference_agreement_z'])<3,
            reference_valid=all(reference_checks.values()),
            initial_family=abs(group.get('initial_family_comparison',{}).get('z',0))<3,
            half_drift=abs(final['half_drift']['z'])<3,
            seed_heterogeneity=final['seed_anova'] is None or final['seed_anova']['p']>=.001,
            empirical_vs_block=.6<final['variance_ratio_empirical_to_batch128']<1.6,
            largest_observed_tail=group['tails']['largest_squared_deviation_fraction']<.05)
        group['calibration_pass']=all(group['calibration_checks'].values())
        group['time_to_precision_diagnostic']=group['time_to_precision'].copy()
        if not group['calibration_pass']:
            group['time_to_precision']={str(delta):None for delta in result['precision_targets']}

if len(result['methods'])==4 and all(len(g['seeds'])==8 for g in result['methods'].values()):
    rng = np.random.default_rng(6911)
    idx = rng.integers(256, size=(6000,256))
    costs, batch_costs = {}, {}
    for method in METHODS:
        main = sorted([r for r in records if r['kind']=='main' and r['method']==method],key=lambda r:r['seed'])
        means=[]
        for r in main:
            with np.load(r['trace']) as trace:
                means.extend(trace['energy'][:,:,0].real.mean(axis=0))
        means=np.array(means)
        t=result['methods'][method]['final']['energy_seconds']
        costs[method]=np.var(means[idx],axis=1,ddof=1)/256*t
        gamma=np.array(result['methods'][method]['final']['batch128_per_walker_gamma'])
        length=max(map(int,result['methods'][method]['curves']))
        batch_costs[method]=np.mean(gamma[idx],axis=1)*t/(length*256)
    for method in METHODS:
        group=result['methods'][method]
        baseline=result['methods']['continuous']['final']['empirical_cost']
        group['empirical_relative_efficiency']=baseline/group['final']['empirical_cost']
        group['efficiency_walker_bootstrap_range']=np.quantile(costs['continuous']/costs[method],[.025,.975]).tolist()
        baseline_batch=result['methods']['continuous']['final']['batch128_cost']
        group['batch128_relative_efficiency']=baseline_batch/group['final']['batch128_cost']
        group['batch128_efficiency_walker_bootstrap_range']=np.quantile(batch_costs['continuous']/batch_costs[method],[.025,.975]).tolist()
        if 'calibration_pass' in group:
            group['efficiency_claim_allowed']=group['calibration_pass'] and result['methods']['continuous']['calibration_pass']

if not args.partial:
    assert counts==dict(fine=8,main=32,reference=4),counts
    assert len(metas)==4 and all(m['final_state']=='successful' and m['checkpoint_unchanged'] for m in metas)
    assert len(set(m['checkpoint_sha256'] for m in metas))==1
    assert hashlib.sha256(Path(metas[0]['checkpoint']).read_bytes()).hexdigest()==metas[0]['checkpoint_sha256']
    assert all(set(g['rank_method_counts'].values())=={2} for g in result['methods'].values())
    for r in records:
        assert r['final_state']=='successful'
    result['verification']=dict(expected_cases=44,all_workers_successful=True,
        device_rotation_balanced=True,checkpoint_unchanged=True)
(output/('summary_partial.json' if args.partial else 'summary.json')).write_text(json.dumps(result,indent=2))
print('ANALYSIS',counts,flush=True)
for method,g in result['methods'].items():
    r=g['final'];print(method,len(g['seeds']),r['mean'],r['independent_walker_se'],r['ac']['se'],r['rhat'],flush=True)
