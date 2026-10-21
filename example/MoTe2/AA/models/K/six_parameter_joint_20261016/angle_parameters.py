"""Standalone parameter evaluator for the exported mixed-function candidate."""
import json
from pathlib import Path
import numpy as np

class AngleParameters:
    def __init__(self,directory=None):
        root=Path(directory) if directory is not None else Path(__file__).resolve().parent
        self.record=json.loads((root/'ANGLE_FUNCTION.json').read_text())
        if self.record['family']!='mixed':raise ValueError('This evaluator exports the mixed family only')
        nodes=np.array(self.record['nodes_deg']);values=np.array(self.record['node_values_eV'])
        self.coefficients=[np.linalg.solve(self.basis(nodes,j),values[j]) for j in range(5)]
        self.a=float(self.record['variables_scaled'][0]*self.record['scales'][0])

    @staticmethod
    def basis(theta,j):
        r=np.sin(np.deg2rad(np.atleast_1d(theta))/2)/np.sin(np.deg2rad(3.)/2)
        if j in [0,1]:
            z=r*r/(1+r*r);return np.c_[np.ones_like(r),z,z*z]
        if j==3:return np.c_[np.ones_like(r),r*r,r**4]
        return np.c_[np.ones_like(r),r,r*r]

    def parameters(self,theta_deg):
        """Return [a_h (eV Å²), Vcos, Vsin, w1, V2, w2 (all eV)]."""
        return np.r_[self.a,[float(self.basis(theta_deg,j)[0]@co) for j,co in enumerate(self.coefficients)]]

    def physical_parameters(self,theta_deg):
        p=self.parameters(theta_deg)
        return {'mass_over_me':3.80998212/p[0],'V1_meV':np.hypot(p[1],p[2])*1000,
            'phi_deg':np.rad2deg(np.arctan2(p[2],p[1])),'w1_meV':p[3]*1000,
            'V2_meV':p[4]*1000,'w2_meV':p[5]*1000}
