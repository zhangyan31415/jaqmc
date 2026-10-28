"""Fresh-process archive/adapter checks; no fitting, HF or NQS calculation."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import catalog

ROOT=catalog.ROOT
HERE=Path(__file__).resolve().parent


def main(output_dir):
    output_dir.mkdir(parents=True,exist_ok=False)
    models=catalog.records()
    ids=[r['id']for r in models]
    assert len(ids)==len(set(ids))
    assert len({r['material']for r in models})==16
    manifests=json.loads((HERE/'ARCHIVE_FILES_20261029.json').read_text())['files']
    for item in manifests:
        p=ROOT/item['path']
        assert p.stat().st_size==item['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],p
        assert p.stat().st_size<50*1024**2,p
    def check(record):
        if not record.get('hf_api'):
            return dict(id=record['id'],scope='Historical hierarchy; documented APIs, no uniform adapter',uniform_adapter=False)
        angle=min(record['angles_deg'],key=lambda x:abs(float(x)-3.89))
        path=output_dir/(record['id']+'.npz')
        command=[sys.executable,'-I',str(HERE/'catalog.py'),'export',record['id'],
                 '--angle',str(angle),'--k','0','0','--k','0.217','0.139','--output',str(path)]
        env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        env.pop('PYTHONPATH',None)
        result=subprocess.run(command,cwd=output_dir,env=env,capture_output=True,text=True,timeout=180)
        (output_dir/(record['id']+'.log')).write_text(result.stdout+'\n'+result.stderr)
        assert result.returncode==0,(record['id'],result.stderr)
        with np.load(path,allow_pickle=False)as z:
            h=z['H'];B=z['B'];points=z['k_fractional']
            assert h.shape[0]==2 and h.shape[1]==h.shape[2]
            assert np.isfinite(h).all()
            error=float(np.max(abs(h-h.conj().transpose(0,2,1))))
            assert error<1e-9,(record['id'],error)
            assert np.allclose(z['k_cartesian'],points@B,atol=1e-13)
            info=json.loads(str(z['metadata_json']))
            assert info['id']==record['id']
            assert info['matrix_carrier'] is not None
        # No-overwrite behavior matters for large HF input batches.
        repeated=subprocess.run(command,cwd=output_dir,env=env,capture_output=True,text=True,timeout=180)
        assert repeated.returncode!=0 and 'Refusing to overwrite' in repeated.stderr
        return dict(id=record['id'],uniform_adapter=True,angle_deg=angle,shape=list(h.shape),
                    Hermitian_max_eV=error,subprocess_exit=0,existing_output_preserved=True,
                    carrier=info['matrix_carrier'])
    with ThreadPoolExecutor(4)as pool:
        checks=list(pool.map(check,models))
    # README link checks are restricted to the generated/current indexes, not
    # historical reports retaining their original provenance-relative links.
    link_count=0
    for path in [ROOT/'README.md',HERE/'MODELS.md',HERE/'README.md',*[ROOT/r['material']/'README.md'for r in models]]:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://'in target or target.startswith('#'):continue
            target=target.split('#')[0]
            if not target:continue
            if (path.parent/target).resolve()==(HERE/'VERIFICATION.json').resolve():
                # This report is written after all checks, including links.
                continue
            assert (path.parent/target).exists(),(path,target)
            link_count+=1
    result=dict(status='PASSED',scope='Portable one-body archive and uniform export interfaces only; no new fit, HF or NQS validation',
                material_count=16,model_version_count=len(models),uniform_matrix_adapter_count=sum(r['uniform_adapter']for r in checks),
                hashed_files_verified=len(manifests),local_index_links_checked=link_count,
                fresh_processes=True,isolated_python=True,source_repository_on_PYTHONPATH=False,
                process_layout='4 independent package subprocesses; BLAS/OMP1 each',
                environment_prefix=sys.prefix,cwd=os.getcwd(),output_dir=str(output_dir),checks=checks)
    (HERE/'VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k!='checks'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    main(args.output_dir.resolve())
