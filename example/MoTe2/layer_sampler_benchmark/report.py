"""Render only verified numerical results; absent/failed gates stay explicit."""
from pathlib import Path
import argparse,json,csv,numpy as np
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();b=a.base
s=json.loads((b/'production/summary.json').read_text());selection=json.loads((b/'frozen/selection.json').read_text())
ref=s['combined_reference'];rows=[]
labels=dict(discrete_flip='离散翻层',conditional='条件抽样',wolff_corrected='修正 Wolff',continuous='连续自旋')
for name,g in s['methods'].items():
    for n,r in g['precision_grid'].items():
        r['absolute_error_95_bound']=abs(r['mean']-ref['mean'])+1.96*np.hypot(r['se_range'][1],ref['se'])
    accuracy={}
    for target in s['precision_targets']:
        grid=sorted((int(n),r) for n,r in g['precision_grid'].items())
        passed=[r['absolute_error_95_bound']<=target for _,r in grid]
        index=next((i for i in range(len(grid)) if all(passed[i:])),None)
        if index is not None and g['calibration_pass']:
            n,r=grid[index];accuracy[str(target)]=dict(records_per_chain=n,
                gpu_seconds=r['energy_seconds'],with_burn_gpu_seconds=r['energy_seconds']+g['burn_seconds'],
                error_bound=r['absolute_error_95_bound'])
        else:accuracy[str(target)]=None
    g['time_to_accuracy_95']=accuracy
    final=g['final']
    for target,cross in g['time_to_precision'].items():
        rows.append(dict(method=name,standard_error_target=float(target),
            gpu_seconds='' if cross is None else cross.get('energy_seconds',cross.get('seconds','')),
            with_burn_gpu_seconds='' if cross is None else cross['with_burn_seconds'],
            calibrated=g['calibration_pass']))
(b/'production/summary_with_accuracy.json').write_text(json.dumps(s,indent=2))
lines=['# 18 空穴、ν=-2/3、2° MoTe₂：冻结神经网络的采样比较','',
 '本报告由已完成的数值文件生成。收敛指所述网络架构内的数值优化平台；没有证明精确基态或 FCI 拓扑。',
 '',f"共同 checkpoint SHA256：`{selection['checkpoint_sha256']}`。",
 f"独立参考能量：{ref['mean']:.8f} ± {ref['se']:.8f} meV/空穴（1 SE）。",'',
 '实测 GPU·秒为各独立链的采样与局域能量计算时间之和；预热、编译和额外关联观测分别记录。',
 '首次达标时间要求后续所有已测前缀继续达标，并通过参考、分块、初态和链混合检查；不是可选停止保证。','',
 '| 方法 | 能量 / meV·空穴⁻¹ | 1 SE | 校验通过 | 相对连续法效率（95%区间） |',
 '|---|---:|---:|---|---|']
for name,g in s['methods'].items():
    f=g['final'];ci=g['efficiency_walker_bootstrap_range']
    efficiency=f"{g['empirical_relative_efficiency']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" if g['efficiency_claim_allowed'] else '未认证'
    lines.append(f"| {labels[name]} | {f['mean']:.7f} | {f['independent_walker_se']:.7f} | {g['calibration_pass']} | {efficiency} |")
lines+=['','| 目标标准误 / meV·空穴⁻¹ | 离散翻层 GPU·秒（含预热） | 条件抽样 | 修正 Wolff | 连续自旋 |','|---:|---:|---:|---:|---:|']
for target in s['precision_targets']:
    vals=[]
    for name in labels:
        c=s['methods'][name]['time_to_precision'].get(str(target))
        vals.append('未认证达标' if c is None else f"{c['energy_seconds']:.1f}（{c['with_burn_seconds']:.1f}）")
    lines.append('| '+str(target)+' | '+' | '.join(vals)+' |')
lines+=['','| 95%绝对误差界目标 / meV·空穴⁻¹ | 离散翻层 GPU·秒（含预热） | 条件抽样 | 修正 Wolff | 连续自旋 |',
    '|---:|---:|---:|---:|---:|']
for target in s['precision_targets']:
    vals=[]
    for name in labels:
        c=s['methods'][name]['time_to_accuracy_95'][str(target)]
        vals.append('未认证达标' if c is None else f"{c['gpu_seconds']:.1f}（{c['with_burn_gpu_seconds']:.1f}）")
    lines.append('| '+str(target)+' | '+' | '.join(vals)+' |')
lines+=['', '优化平台候选与最终独立验证：', '',
 '| 种子 | 平台步数 | 原验证能量 | 新验证能量 | 新验证 SE | 混合检查 |',
 '|---|---:|---:|---:|---:|---|']
for c in selection['candidates']:
    fresh=next(v for v in selection['cross_replica_final_holdouts'] if v['path']==c['path'])
    lines.append(f"| {Path(c['path']).parent.name} | {c['step']} | {c['mean']:.7f} | {fresh['mean']:.7f} | {fresh['se']:.7f} | {fresh['mixing_pass']} |")
lines+=['',f"四个训练副本的认证状态：`{selection['replica_status']}`。未通过副本不用于冻结选择。"]
lines+=['', '层与电荷关联（已扣除平均模的平方）：', '',
 '| 方法 | 层极化 | 连接层 Szz(0) | 相对独立层涨落 | 最低非零q电荷S |',
 '|---|---:|---:|---:|---|']
for name,g in s['methods'].items():
    z=g['structure']
    lines.append(f"| {labels[name]} | {z['polarization']:.5f} | {z['connected_spin_structure'][0]:.5f} | {z['collective_zero_q_ratio']:.5f} | {z['connected_charge_structure'][1:]} |")
failed={name:[key for key,value in g['calibration_checks'].items() if not value] for name,g in s['methods'].items() if not g['calibration_pass']}
if failed:
    lines+=['',f'仍未通过的校验：`{failed}`。这些方法的效率排序和达标时间暂不认证。']
lines+=[
 '', '方法限制：原胞周期单体特征、超胞周期相对坐标特征、32×3/16×3 网络、一个基础行列式及同一 Gamma 投影的27个平移副本。',
 '生产哈密顿量保留连续空间双层势、隧穿、谷动量修正和完整2D Ewald相互作用；平面波截断只用于固定初始化基底。',
 'Wolff保持精确目标分布的接受率修正；代理模型及其独立调参记录见 frozen/wolff_tuning.json。',
 '层团簇算法是否有优势需结合层关联与电荷关联分别判断，不能由填充或相互作用强度单独推断。']
if (b/'energy_precision_vs_time.png').exists():
    lines+=['',f'![Energy precision versus measured time]({b}/energy_precision_vs_time.png)']
(b/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
with (b/'production/time_to_precision.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('REPORT',b/'REPORT_ZH.md',flush=True)
