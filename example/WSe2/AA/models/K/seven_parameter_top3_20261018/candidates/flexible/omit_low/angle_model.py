"""Portable seven-term model with the selected angle-function coefficients."""
import json
from pathlib import Path
import numpy as np
from model import response7,HBAR2

class AngleModel:
    def __init__(self,configuration=None):
        path=Path(configuration) if configuration else Path(__file__).with_name('FUNCTION.json')
        self.config=json.loads(path.read_text())
    def coordinate(self,theta):
        r=np.sin(np.deg2rad(theta)/2)/np.sin(np.deg2rad(3.)/2)
        return {'r':r,'r2':r*r,'z':r*r/(1+r*r)}[self.config['coordinate']]
    def parameters(self,theta):
        lo,hi=self.config['function_domain_deg']
        u=(self.coordinate(theta)-self.coordinate(lo))/(self.coordinate(hi)-self.coordinate(lo))
        b=np.array([1-u,u]) if self.config['degree']==1 else np.array([(1-u)**2,2*u*(1-u),u*u])
        return np.r_[self.config['a_h_eV_A2'],np.array(self.config['potential_controls_eV'])@b,self.config['lambda_eV_A']]
    def physical_parameters(self,theta):
        p=self.parameters(theta)
        return {'mass_over_me':HBAR2/p[0],'V1_meV':np.hypot(p[1],p[2])*1000,'phi_deg':np.rad2deg(np.arctan2(p[2],p[1])),
            'w1_meV':p[3]*1000,'V2_meV':p[4]*1000,'w2_meV':p[5]*1000,'lambda_eV_A':p[6]}
    def geometry(self,theta,nq=169):
        if theta<=0:raise ValueError('Positive angle required')
        ref=self.config['geometry_reference'];angles=self.config['geometry_angles_deg'];scales=self.config['normalised_reciprocal_scales']
        ratio=np.sin(np.deg2rad(theta)/2)/np.sin(np.deg2rad(ref['angle_deg'])/2)*np.interp(theta,angles,scales)/scales[0]
        b=np.array(ref['reciprocal'])*ratio;kap=np.array(ref['kappas'])*ratio
        ij=np.array([(i,j) for i in range(-35,36) for j in range(-35,36)])
        candidates=kap[0]-ij@b;norm=np.linalg.norm(candidates,axis=1);radius=np.sort(norm)[nq-1];q1=candidates[norm<=radius+1e-10]
        return {'q1':q1,'q2':-q1,'reciprocal':b,'kappas':kap,'fractional_to_model':np.array(ref['fractional_to_model'])*ratio}
    def hamiltonian(self,theta,k,valley='K',nq=169):
        if valley not in ['K','Kprime']:raise ValueError(valley)
        g=self.geometry(theta,nq);k=np.asarray(k) if valley=='K' else -np.asarray(k)
        h=np.einsum('pij,p->ij',response7(k,g['q1'],g['q2'],g['reciprocal'],g['kappas'],theta)[0],self.parameters(theta))
        return h if valley=='K' else h.conj()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--angle',required=True,type=float);a=p.parse_args()
    print(json.dumps(AngleModel().physical_parameters(a.angle),indent=2))
