"""Select on pilot; evaluate frozen choices on independent holdout chains."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
from cadence_stats import replicate_bootstrap,bootstrap_indices,stable_crossings,select_cadences
sys.path.insert(0,str(Path(__file__).resolve().parent/'source'))
from mcstats import summary,seed_anova,independent_mean_se

METHODS=['discrete_flip','conditional','wolff_corrected','continuous']
LABELS={'discrete_flip':'离散翻层','conditional':'条件抽样','wolff_corrected':'修正 Wolff','continuous':'连续自旋'}
EXPECTED='6f79f2570485b3202cf40fac72c20b230ca286108e1faed5c0bb63a44f6b2423'
TARGETS=[.02,.01,.005,.002]


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def inspect_group(records,reference,repeats,curves=False):
    records=sorted(records,key=lambda r:r['seed']);energies=[];obs=[];timing=[]
    for r in records:
        assert r['checkpoint_sha256']==EXPECTED and r['kinetic_mode']=='forward_laplacian'
        assert r['final_state']=='successful' and r['measurements']>=2048
        with np.load(r['trace']) as t:
            energies.append(t['energy']);obs.append(t['observables'][:,:,[0,15,19]]);timing.append(t['times'])
    energy=np.stack(energies,axis=1);observable=np.stack(obs,axis=1)
    x=energy[:,:,:,0].real;im=energy[:,:,:,0].imag;T,S,W=x.shape;M=S*W
    assert np.isfinite(energy).all() and np.isfinite(observable).all()
    diag=summary(x);means=x.mean(axis=0);se=independent_mean_se(x)
    seconds=float(sum(t[:,:2].sum() for t in timing));burn=sum(r['burn_seconds'] for r in records)
    boot_se2=replicate_bootstrap(means,repeats,71022);boot_cost=boot_se2*seconds
    obdiag={name:summary(observable[:,:,:,i]) for i,name in enumerate(['layer','charge_Sq1','charge_ReRhoq1'])}
    half=x[T//2:].mean(axis=0)-x[:T//2].mean(axis=0)
    drift_z=float(half.mean()/max(half.std(ddof=1)/np.sqrt(M),1e-30))
    im_se=independent_mean_se(im)
    ref_z=float((x.mean()-reference['mean'])/np.hypot(se,reference['se']))
    families={}
    for family in sorted({r['initial_family'] for r in records}):
        ix=[i for i,r in enumerate(records) if r['initial_family']==family]
        a=x[:,ix];families[family]=dict(mean=float(a.mean()),se=independent_mean_se(a))
    family_z=0.
    if len(families)==2:
        a,b=families.values();family_z=(a['mean']-b['mean'])/np.hypot(a['se'],b['se'])
    blocks={str(b):diag['block_se'][str(b)] for b in [64,128,256] if str(b) in diag['block_se']}
    checks=dict(energy_rhat=diag['rhat']['rank']<1.01,
                invariant_charge_rhat=all(obdiag[k]['rhat']['rank']<1.01 for k in ['charge_Sq1','charge_ReRhoq1']),
                layer_rhat=obdiag['layer']['rhat']['rank']<1.01,
                block_vs_replicate=.8<diag['block_se']['128']/se<1.25,
                block_plateau=max(blocks.values())/min(blocks.values())<1.25,
                imaginary_mean=abs(im.mean())<4*im_se,
                reference_agreement=abs(ref_z)<4,half_drift=abs(drift_z)<4,initial_family_agreement=abs(family_z)<4)
    checks={name:bool(ok) for name,ok in checks.items()}
    result=dict(method=records[0]['method'],cadence=records[0]['cadence_sweeps'],seeds=[r['seed'] for r in records],
        seed_count=S,walkers_per_seed=W,records_per_chain=T,mean=float(x.mean()),se=se,
        se_95_interval=np.sqrt(np.quantile(boot_se2,[.025,.975])).tolist(),
        cost=float(se**2*seconds),cost_95_interval=np.quantile(boot_cost,[.025,.975]).tolist(),
        timed_gpu_seconds=seconds,burn_gpu_seconds=burn,compile_gpu_seconds=sum(r['compilation_seconds'] for r in records),
        sampler_gpu_seconds=sum(r['sampler_seconds'] for r in records),energy_gpu_seconds=sum(r['energy_seconds'] for r in records),
        observable_gpu_seconds=sum(r['observable_seconds'] for r in records),
        seconds_per_batch32=seconds/(T*S),real_variance=float(x.var()),g_energy=2*diag['ac']['tau'],
        imaginary_mean=float(im.mean()),imaginary_se=im_se,reference_z=ref_z,half_drift_z=drift_z,
        family_z=float(family_z),initial_families=families,seed_anova=seed_anova(x),diagnostics=diag,
        observable_diagnostics=obdiag,checks=checks,passed=all(checks.values()),
        acceptance=np.mean([r['acceptance'] for r in records],axis=0).tolist(),
        time_ci_scope='Whole-walker statistical uncertainty conditional on measured timing and hardware; excludes queue time',
        asymptotic_time_to_precision={str(d):dict(hot_gpu_seconds=float(se**2*seconds/d**2),
            hot_95_interval=(np.quantile(boot_cost,[.025,.975])/d**2).tolist(),
            with_burn_gpu_seconds=float(se**2*seconds/d**2+burn),
            with_burn_95_interval=(np.quantile(boot_cost,[.025,.975])/d**2+burn).tolist(),
            scope='Asymptotic cost scaling estimate, not an observed threshold crossing') for d in TARGETS})
    if curves:
        prefixes=list(range(64,T+1,64));flat=x.reshape(T,M);cumulative=np.cumsum(flat,axis=0)
        idx=bootstrap_indices(S,W,repeats,71022);boot_se=[];point_se=[];upper_se=[];times=[]
        for n in prefixes:
            a=cumulative[n-1]/n
            boot=np.std(a[idx],axis=1,ddof=1)/np.sqrt(M)
            boot_se.append(boot);point_se.append(float(a.std(ddof=1)/np.sqrt(M)))
            upper_se.append(float(np.quantile(boot,.975)));times.append(sum(float(t[:n,:2].sum()) for t in timing))
        boot_se=np.stack(boot_se,axis=1);result['precision_grid']=dict(prefixes=prefixes,se=point_se,se_upper=upper_se,hot_gpu_seconds=times)
        crossings={}
        for d in TARGETS:
            ci=stable_crossings(boot_se,times,d)
            point=stable_crossings(np.asarray(point_se)[None,:],times,d)['crossing_seconds'][0]
            conservative=stable_crossings(np.asarray(upper_se)[None,:],times,d)['crossing_seconds'][0]
            crossings[str(d)]=dict(observed_hot_gpu_seconds=point,
                observed_with_burn_gpu_seconds=point+burn if point is not None else None,
                conservative_upper_se_with_burn_gpu_seconds=conservative+burn if conservative is not None else None,
                bootstrap_hot_95_interval=ci['interval_95_seconds'],
                bootstrap_with_burn_95_interval=[v+burn if v is not None else None for v in ci['interval_95_seconds']],
                bootstrap_right_censored_fraction=ci['censored_fraction'],
                bootstrap_left_censored_at_first_grid_fraction=ci['left_censored_at_first_grid_fraction'],
                observed_crossing_at_first_grid=bool(point is not None and point==times[0]),
                scope='Retrospective stable crossing on measured grid; not an optional-stopping guarantee')
        result['observed_precision_crossings']=crossings
    return result,boot_cost


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--phase',choices=['pilot','holdout'],required=True);ap.add_argument('--bootstrap',type=int,default=4000);args=ap.parse_args()
    original=json.loads((args.base/'production/summary_with_accuracy.json').read_text())
    ref=original['combined_reference']
    reference=dict(mean=ref['mean'],se=ref['se'])
    rows={};boots={};seen=set()
    for path in sorted((args.out/args.phase).glob('k*/records/*.json')):
        r=json.loads(path.read_text());key=(r['method'],r['cadence_sweeps'],r['seed'])
        if key in seen:raise ValueError(('duplicate case',key))
        seen.add(key);rows.setdefault((key[0],key[1]),[]).append(r)
    expected=8 if args.phase=='pilot' else 12
    results={m:{} for m in METHODS}
    for (m,k),records in sorted(rows.items()):
        if len(records)!=expected:raise ValueError(('incomplete',m,k,len(records),expected))
        result,boot=inspect_group(records,reference,args.bootstrap,args.phase=='holdout')
        results[m][str(k)]=result;boots[m,k]=boot
        print('GROUP',args.phase,m,k,result['cost'],result['se'],result['checks'],flush=True)
    if args.phase=='pilot':
        assert all(set(v)=={'5','10','20','40'} for v in results.values())
        selected=select_cadences(results);selection=dict(checkpoint_sha256=EXPECTED,selected_cadences=selected,
            selection_metric='Measured sampler+forward-energy GPU seconds times independent-walker MCSE squared',
            stage='Frozen before independent holdout',pilot_seed_count=8,selection_probabilities={},relative_to_k10={},
            pilot_diagnostic_failures={f'{m}/k{k}':[name for name,ok in g['checks'].items() if not ok] for m,cases in results.items() for k,g in cases.items() if not g['passed']},
            selection_caveat='Finite pilot point optimum; any pilot mixing flags require successful independent holdout before interpretation')
        for m in METHODS:
            ks=sorted(map(int,results[m]));draw=np.stack([boots[m,k] for k in ks]);winners=np.argmin(draw,axis=0)
            selection['selection_probabilities'][m]={str(k):float(np.mean(winners==i)) for i,k in enumerate(ks)}
            selection['relative_to_k10'][m]={str(k):dict(cost_ratio=results[m][str(k)]['cost']/results[m]['10']['cost'],
                interval_95=np.quantile(boots[m,k]/boots[m,10],[.025,.975]).tolist()) for k in ks}
        if (args.out/'selection.json').exists():
            previous=json.loads((args.out/'selection.json').read_text())
            assert previous==selection,'Frozen selection cannot be silently changed'
        atomic_json(args.out/'pilot_summary.json',dict(methods=results,reference=reference))
        atomic_json(args.out/'selection.json',selection)
        print('SELECTION_FROZEN',json.dumps(selection),flush=True)
    else:
        selection=json.loads((args.out/'selection.json').read_text());chosen=selection['selected_cadences'];comparisons={}
        assert all(set(map(int,results[m]))=={10,chosen[m]} for m in METHODS)
        for m in METHODS:
            k=chosen[m];kc=chosen['continuous'];group=results[m][str(k)];cont=results['continuous'][str(kc)]
            comparisons[m]=dict(cadence=k,cost_ratio_to_continuous=group['cost']/cont['cost'],
                ratio_95_interval=np.quantile(boots[m,k]/boots['continuous',kc],[.025,.975]).tolist(),
                cost_ratio_to_own_k10=group['cost']/results[m]['10']['cost'],
                ratio_to_own_k10_95=np.quantile(boots[m,k]/boots[m,10],[.025,.975]).tolist())
        passed=all(v['passed'] for m in results.values() for v in m.values())
        output=dict(checkpoint_sha256=EXPECTED,selection=selection,methods=results,comparisons=comparisons,
                    reference=reference,passed=passed,scope='Frozen-checkpoint sampling precision, not ground-state accuracy')
        atomic_json(args.out/'holdout_summary.json',output)
        lines=['# 固定 MoTe₂ checkpoint：forward Laplacian 与测量间隔实测','',
            '18空穴、27莫尔胞、2°。所有能量精度均为 meV/空穴的1σ MCSE；不是精确基态误差。',
            '时间是实测累计 GPU·秒。置信区间保留整条独立链，条件于本次测得的硬件计时。',
            '先用独立 pilot 选择间隔，再用全新种子比较；cadence=10 同时作独立基线。','',
            '| 方法 | 选中间隔 | 能量 | MCSE | 相对连续法成本 [95%区间] | 相对自身间隔10成本 [95%区间] |',
            '|---|---:|---:|---:|---|---|']
        for m in METHODS:
            c=comparisons[m];g=results[m][str(chosen[m])];a,b=c['ratio_95_interval'];d,e=c['ratio_to_own_k10_95']
            lines.append(f"| {LABELS[m]} | {chosen[m]} | {g['mean']:.8f} | {g['se']:.6f} | {c['cost_ratio_to_continuous']:.3f} [{a:.3f}, {b:.3f}] | {c['cost_ratio_to_own_k10']:.3f} [{d:.3f}, {e:.3f}] |")
        lines+=['','以下为实测序列中标准误持续低于目标的达标时间，含预热；括号是 bootstrap 95%区间。',
            '达标规则是回顾性统计，不提供可选停止保证。未跨过阈值的 bootstrap 重复被右删失，不能外推成已测时间。','',
            '| 目标 MCSE | 离散翻层 | 条件抽样 | 修正 Wolff | 连续自旋 |','|---:|---|---|---|---|']
        def fmt(point,bounds):
            val='未达到' if point is None else f'{point:.0f}'
            ci=['右删失' if v is None else f'{v:.0f}' for v in bounds]
            return f'{val} [{ci[0]}, {ci[1]}]'
        for target in TARGETS:
            cells=[]
            for m in METHODS:
                crossing=results[m][str(chosen[m])]['observed_precision_crossings'][str(target)]
                cells.append(fmt(crossing['observed_with_burn_gpu_seconds'],crossing['bootstrap_with_burn_95_interval']))
            lines.append('| '+str(target)+' | '+' | '.join(cells)+' |')
        lines+=['',f'全部预设混合/误差检查通过：{passed}。具体检查、谱估计、分块平台、删失比例见 holdout_summary.json。',
            '编译、pilot 调参、额外诊断和 I/O 不计入上述生产时间；它们另行记录。',
            '同一 GPU 内没有并发工作进程。多 GPU 累计时间不能直接称为墙钟时间。',
            '选中间隔是有限候选中的 pilot 点估计最优；selection_probabilities 是 bootstrap 选择频率，不是真最优的概率。',
            '首次64-record网格已达标时，耗时区间受网格分辨率限制，并不精确定位真正的最短时间。',
            '渐近成本除以目标精度平方得到的时间另存于 JSON，明确标为估计，不混入实测达标表。','']
        (args.out/'REPORT_ZH.md').write_text('\n'.join(lines))
        atomic_json(args.out/'SCIENTIFIC_STATUS.json',dict(passed=passed,failed_checks={f'{m}/k{k}':[name for name,ok in g['checks'].items() if not ok] for m,v in results.items() for k,g in v.items() if not g['passed']}))
        print('HOLDOUT_ANALYSIS_DONE',passed,flush=True)
        if not passed:raise SystemExit(3)

if __name__=='__main__':main()
