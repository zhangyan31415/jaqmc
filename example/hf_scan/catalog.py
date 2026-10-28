"""List archived one-body models and export matrices for an external HF code.

The export does not choose interactions, project a density operator, or solve HF.
Only NumPy and the selected model's documented dependencies are required.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def records():
    result = []
    for path in sorted(ROOT.glob('*/**/MODEL_INFO.json')):
        data = json.loads(path.read_text())
        data['directory'] = str(path.parent.relative_to(ROOT))
        data['descriptor'] = str(path.relative_to(ROOT))
        result.append(data)
    return result


def choose(model_id):
    matches = [r for r in records() if r['id'] == model_id]
    if len(matches) != 1:
        raise ValueError(f'Expected one model with id={model_id!r}; found {len(matches)}')
    return matches[0]


def export_worker(args):
    import numpy as np

    record = choose(args.model)
    output=Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    api = record.get('hf_api')
    if not api:
        raise ValueError('This archive has no uniform matrix adapter; use its documented model API.')
    folder = ROOT / record['directory']
    module_path = folder / api['module']
    if not module_path.suffix:
        module_path=module_path.with_suffix('.py')
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location('_hf_selected_model', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hamiltonian = getattr(module, api.get('hamiltonian', 'hamiltonian'))
    basis = getattr(module, api.get('reciprocal_basis', 'reciprocal_basis'))
    angles = record.get('angles_deg', [])
    theta = args.angle
    if not np.isfinite(theta) or not angles or min(abs(float(x) - theta) for x in angles) > 0.006:
        raise ValueError('Use a catalogued geometry angle; arbitrary geometry extrapolation is not certified.')
    B = np.asarray(basis(theta), dtype=float)
    if B.shape != (2, 2) or not np.all(np.isfinite(B)) or abs(np.linalg.det(B)) < 1e-15:
        raise ValueError('The adapter must provide a nonsingular 2x2 reciprocal basis in inverse Angstrom.')
    if args.nk is not None:
        if args.nk < 1:
            raise ValueError('--nk must be positive')
        points = np.array([[i/args.nk,j/args.nk] for i in range(args.nk) for j in range(args.nk)])
    else:
        points = np.array(args.k or [[0.,0.]],dtype=float)
    first = np.asarray(hamiltonian(theta, points[0]), dtype=complex)
    if first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError('The adapter must return a square one-body Hamiltonian.')
    estimated = len(points)*first.size*16
    if estimated > 512*1024**2:
        raise ValueError('Dense export exceeds 512 MiB; export k batches rather than silently allocating a large cube.')
    matrices=[]
    for k in points:
        h=np.asarray(hamiltonian(theta,k),dtype=complex)
        if h.shape != first.shape or not np.all(np.isfinite(h)):
            raise ValueError('Hamiltonian shape or finite-value check failed.')
        if not np.allclose(h,h.conj().T,atol=1e-10,rtol=1e-10):
            raise ValueError('Hamiltonian is not Hermitian at the requested k.')
        matrices.append(h)
    matrices=np.array(matrices)
    metadata={**record,'export_angle_deg':theta,'matrix_carrier':api.get('carrier',record.get('carrier_convention')),
              'export_k_convention':'k_cartesian = k_fractional @ B; B rows in inverse Angstrom',
              'scope':'Unmodified one-body matrices, no interacting HF or density-vertex approximation supplied.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(theta_deg=theta,k_fractional=points,k_cartesian=points@B,B=B,H=matrices,
                 metadata_json=json.dumps(metadata,ensure_ascii=False))
    if args.eigenvalues:
        payload['eigenvalues_eV']=np.linalg.eigvalsh(matrices)
    with output.open('wb')as stream:
        np.savez_compressed(stream,**payload)
    print(json.dumps(dict(model=record['id'],path=str(output),matrix_shape=list(matrices.shape),
                         energy_unit='eV',carrier=metadata['matrix_carrier']),ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    listing=sub.add_parser('list',help='List all model versions, including diagnostics')
    listing.add_argument('--material')
    listing.add_argument('--json',action='store_true')
    show=sub.add_parser('show',help='Show the complete model descriptor')
    show.add_argument('model')
    export=sub.add_parser('export',help='Export one-body matrices in a fresh Python process')
    export.add_argument('model')
    export.add_argument('--angle',type=float,required=True)
    export.add_argument('--k',type=float,nargs=2,action='append',metavar=('K1','K2'))
    export.add_argument('--nk',type=int,help='Uniform fractional grid in [0,1)^2; instead of --k')
    export.add_argument('--eigenvalues',action='store_true')
    export.add_argument('--output',required=True)
    export.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.action=='list':
        data=[r for r in records() if not args.material or r.get('material','').lower()==args.material.lower()]
        if args.json:
            print(json.dumps(data,ensure_ascii=False,indent=2))
        else:
            for r in data:
                print(f"{r['id']} | {r.get('material')} {r.get('stacking')}/{r.get('valley')} | "
                      f"parameters={r.get('physical_parameters')} | {r.get('status')} | {r['directory']}")
    elif args.action=='show':
        print(json.dumps(choose(args.model),ensure_ascii=False,indent=2))
    elif args.worker:
        if args.k and args.nk is not None:
            raise ValueError('Choose --k or --nk, not both.')
        export_worker(args)
    else:
        # Each package may use conventional module names (model/kernel). Never
        # reuse their sys.modules entries across material imports in one process.
        subprocess.run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],check=True)


if __name__=='__main__':
    main()
