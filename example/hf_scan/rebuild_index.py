"""Build the dated model/HF handoff index from portable package descriptors."""
from pathlib import Path
import csv
import hashlib
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import catalog

ROOT=catalog.ROOT
HERE=Path(__file__).resolve().parent
DATE='2026-10-29'


def tier(record):
    model_id=record['id']
    material=record['material']
    if material=='CdCl2' and record['physical_parameters']==3:
        return 'first_band_candidate','First conduction band; inspect every angle and do not include outside-sector higher bands.'
    if material in ['HfSe2','HfS2'] or (material=='ZrS2' and record['stacking']=='AA'):
        return 'small_angle_candidate','First two conduction pairs at theta <=3.890238169 degrees; examine SOC/source-Q limitations.'
    if material in ['SnS2','SnSe2']:
        return 'exploratory_cutoff_check_required','Small-angle first-two-band comparison; existing enlarged-model-Q sensitivity is significant.'
    if material=='WSe2' and '17angles' in model_id:
        return 'calibrated_candidate_with_wave_limits','17 calibrated angles; Heff wave overlap; 5.09-degree third-band mismatch remains.'
    if material=='InSe' and record['physical_parameters']==10:
        return 'exploratory_small_angle','Small-angle first two pairs; finite fit budget and imperfect high-angle states.'
    if material=='GaSe' and record['physical_parameters']==10:
        return 'exploratory_restricted_angles','Best within roughly2.28--3.15 degrees; all-angle fit error is large.'
    if material=='MoTe2' and '20261026' in model_id:
        return 'exploratory_tradeoff','Compare the energy-selected and better-wavefunction branches; no unique qualified winner.'
    return 'diagnostic_or_historical','Read version-specific status before use; not a default production input.'


def main():
    models=catalog.records()
    ids=[r['id']for r in models]
    assert len(ids)==len(set(ids)),'Duplicate model id'
    required=['id','material','stacking','valley','physical_parameters','angles_deg','status','hf_scope',
              'basis_description','energy_unit','length_unit','k_convention','carrier_convention']
    for r in models:
        assert all(k in r for k in required),(r['id'],set(required)-r.keys())
        r['scan_tier'],r['scan_range_note']=tier(r)
    result=dict(date=DATE,scope='Portable one-body archive and external HF handoff; not an HF solver or material-accuracy recertification',
                material_count=len({r['material']for r in models}),model_version_count=len(models),
                uniform_matrix_adapter_count=sum(bool(r.get('hf_api'))for r in models),models=models)
    (HERE/'MODEL_CATALOG.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (HERE/'MODEL_CATALOG.csv').open('w',newline='',encoding='utf-8-sig')as out:
        writer=csv.writer(out);writer.writerow(['id','material','stacking','valley','edge','physical_parameters','global_coefficients','angles','scan_tier','scan_range_note','directory','uniform_matrix_adapter'])
        for r in models:
            writer.writerow([r.get(k)for k in ['id','material','stacking','valley','edge','physical_parameters','global_coefficients']]+[len(r['angles_deg']),r['scan_tier'],r['scan_range_note'],r['directory'],bool(r.get('hf_api'))])
    files=[];seen=set()
    for r in models:
        folder=ROOT/r['directory']
        for p in sorted(folder.rglob('*')):
            if not p.is_file()or'__pycache__'in p.parts or p.suffix=='.pyc':continue
            relative=str(p.relative_to(ROOT))
            if relative in seen:continue
            seen.add(relative)
            files.append(dict(path=relative,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (HERE/'ARCHIVE_FILES_20261029.json').write_text(json.dumps(dict(date=DATE,files=files),indent=2)+'\n')
    materials=sorted({r['material']for r in models})
    lines=['# Current continuum model packages','',f'Updated {DATE}: {len(materials)} materials, {len(models)} model versions, '+
           f"{result['uniform_matrix_adapter_count']} uniform one-body matrix adapters.",'',
           'Physical parameters per angle and global angle-law coefficients are separate counts. '
           'Diagnostic and historical versions remain visible; inclusion is not an accuracy certificate.','',
           '| Material | Stacking / valley | Current versions | One-body scope |','|---|---|---|---|']
    for material in materials:
        rs=[r for r in models if r['material']==material]
        targets=', '.join(sorted({r['stacking']+'/'+r['valley']for r in rs}))
        links='; '.join(f"[{r['id']}](../{r['directory']}/README.md)"for r in rs)
        scopes='; '.join(sorted({r['scan_tier']for r in rs}))
        lines.append(f'| {material} | {targets} | {links} | {scopes} |')
    lines+=['','## Read before a scan','',
            '- [Commands, conventions and interaction inputs](README.md).',
            '- [Machine-readable model catalogue](MODEL_CATALOG.json) and [CSV](MODEL_CATALOG.csv).',
            '- The old ZrS2 AB six-coefficient target is Gamma with eight internal states per Q. The new ZrS2 AA ten-coefficient target is M1 with one spatial orbital per layer.',
            '- WSe2 seventeen-angle and MoTe2 ten-angle versions are distinct from the historical October18/16 archives.',
            '- No atomic H/S or native-wavefunction cache is required to load the one-body packages. Density vertices and interactions are separate inputs.',
            '- Every new package contains its source-file hashes and numerical replay checks. See [integration verification](VERIFICATION.json) for fresh-process common-adapter checks.']
    (HERE/'MODELS.md').write_text('\n'.join(lines)+'\n')
    for material in materials:
        root=ROOT/material
        path=root/'README.md'
        old=path.read_text()if path.exists()else f'# {material}\n'
        marker='## Current continuum archive (2026-10-29)'
        if marker in old:old=old.split(marker)[0].rstrip()+'\n'
        extra=[marker,'','[All materials and HF handoff](../hf_scan/MODELS.md). Versions below retain their scientific limits.','']
        for r in models:
            if r['material']!=material:continue
            relative=Path(r['directory']).relative_to(material)
            extra.append(f"- [{r['id']}]({relative}/README.md): {r['physical_parameters']} physical parameters, "
                         f"{r.get('global_coefficients')} angle coefficients; {r['status']}.")
        path.write_text(old+'\n'+'\n'.join(extra)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k!='models'},indent=2))


if __name__=='__main__':main()
