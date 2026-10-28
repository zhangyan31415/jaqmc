"""Add descriptors for unchanged historical archives, without changing fits."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    for material in ['SnS2','SnSe2']:
        folder=ROOT/material/'AA/models/M/seven_parameter_constant_mass_20261021'
        catalog=json.loads((folder/'CATALOG.json').read_text())
        info=dict(id=f'{material}_AA_M_seven_parameter_constant_mass_20261021',
            material=material,stacking='AA',valley='M1',edge='CBM',
            physical_parameters=7,global_coefficients=17,
            angles_deg=[r['theta_deg']for r in catalog],
            basis_description='One envelope orbital per layer, component-major plane waves; spectator physical spin is not copied in the orbital matrix.',
            energy_unit='eV',length_unit='Angstrom',k_convention='fractional B; reciprocal vectors are rows in inverse Angstrom',
            carrier_convention='electron',status='ARCHIVED_SMALL_ANGLE_CANDIDATE_WITH_Q_LIMITS',
            hf_scope='Small-angle first-two-band comparison only; full three-band all-angle fit failed and enlarged model-Q sensitivity is material.',
            entrypoint='model.py',reference_type='Native TAPW pair centers and raw fixed-source wave projections',
            hf_api=dict(module='hf_adapter.py',hamiltonian='hamiltonian',reciprocal_basis='reciprocal_basis',carrier='electron'),
            limitations=['Native SOC omitted; spectator spin approximation','Area-two periodic cell or layer-twisted primitive boundary required',
                         'Model-Q sensitivity is not negligible and native-Q convergence not established',
                         'No microscopic interaction density vertex or HF solver',
                         'Fixed-state external NQS timing is not a native training preset'],
            source_files=['COEFFICIENTS.json','CATALOG.json','reports/SOURCE_EXPORT_VERIFICATION.json'],
            hf_scan_recommendation='EXPLORATORY_ONLY; explicit angle/cutoff selection required')
        (folder/'MODEL_INFO.json').write_text(json.dumps(info,indent=2)+'\n')
        (folder/'hf_adapter.py').write_text('''"""Electron one-body matrix, eV; fractional k is with respect to B rows."""
from pathlib import Path
import importlib.util
import numpy as np
_path=Path(__file__).resolve().with_name("model.py")
_spec=importlib.util.spec_from_file_location("_historical_m_model",_path)
_model=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_model)
def hamiltonian(theta_deg,k_fractional):
    return _model.hamiltonian(theta_deg,np.asarray(k_fractional))
def reciprocal_basis(theta_deg):
    record=_model.record(theta_deg)
    return _model.geometry(record["theta_deg"])["B"].copy()
''')
    for material,version,count,global_count,description in [
        ('MoTe2','six_parameter_joint_20261016',6,16,'Historical nine-angle K-valley model; latest ten-angle variants are separate versions.'),
        ('WSe2','seven_parameter_top3_20261018',7,12,'Historical ten-angle seven-term K model; latest seventeen-angle version is separate.')]:
        folder=ROOT/material/'AA/models/K'/version
        if material=='MoTe2':
            cases=folder/'cases'
        else:
            cases=folder/'candidates/strict/models'
        labels=[p.name.split('_',1)[-1]for p in cases.iterdir()]if cases.is_dir()else []
        angles=[]
        for name in labels:
            try:angles.append(float(name))
            except ValueError:pass
        if material=='WSe2':
            configuration=json.loads((folder/'candidates/robust/full/FUNCTION.json').read_text())
            angles=[float(t)for t in configuration['geometry_angles_deg']if t<=5.1]
        info=dict(id=f'{material}_AA_K_{version}',material=material,stacking='AA',valley='K',edge='VBM',
            physical_parameters=count,global_coefficients=global_count,angles_deg=sorted(angles),
            basis_description='One selected-spin envelope orbital per layer; opposite valley follows the documented time-reversal convention.',
            energy_unit='eV',length_unit='Angstrom',k_convention='Source fractional-to-model transform; see archived README, not a uniform adapter.',
            carrier_convention='Source electron/hole APIs; choose explicitly as documented',
            status='HISTORICAL_ARCHIVE',hf_scope=description,entrypoint=None,
            reference_type='Archived Heff model overlap; not a newly certified native-TAPW embedding',
            limitations=['Use current version descriptors for a new scan','No uniform adapter added to this unchanged historical candidate hierarchy'],
            source_files=['README.md'])
        if material=='WSe2':
            info['geometry_angles_deg_including_diagnostic_extrapolation']=configuration['geometry_angles_deg']
        (folder/'MODEL_INFO.json').write_text(json.dumps(info,indent=2)+'\n')
    folder=ROOT/'ZrS2/AB/models/Gamma/six_parameter_20261014'
    cases=folder/'candidates/independent/cases'
    angles=sorted(float(p.name.split('_',1)[1])for p in cases.iterdir()if p.is_dir())
    info=dict(id='ZrS2_AB_Gamma_six_parameter_20261014',material='ZrS2',stacking='AB',valley='Gamma',
        edge='Source upper eight-state Gamma window; inspect source energy and carrier convention',physical_parameters=6,
        global_coefficients=None,angles_deg=angles,
        basis_description='Q-major, layer x two orbital labels x physical spin: eight internal components per Q; two physical layers.',
        energy_unit='eV',length_unit='Angstrom',k_convention='Cartesian inverse Angstrom with the saved coordinate rotation; read each case model.py',
        carrier_convention='Electron Hamiltonian; a1,a2 are matrix kinetic coefficients, not one scalar mass',
        status='HISTORICAL_ARCHIVE_WITH_LARGE_ANGLE_FAILURES',
        hf_scope='Independent case-by-case archive and joint diagnostics; 3.15-degree gap candidate is distinct. Not the new ZrS2 AA/M model.',
        entrypoint=None,reference_type='Native TAPW with frozen transported source basis',
        limitations=['Not an analytic global angle law or a common scalar mass','Large-angle wavefunction failures retained',
                     'No uniform adapter added to nested independent/joint case archives','No microscopic interaction vertex supplied'],
        source_files=['README.md','candidates/independent/parameters.csv'])
    (folder/'MODEL_INFO.json').write_text(json.dumps(info,indent=2)+'\n')
    print('Registered five existing historical material archives; fits and reports unchanged.')


if __name__=='__main__':main()
