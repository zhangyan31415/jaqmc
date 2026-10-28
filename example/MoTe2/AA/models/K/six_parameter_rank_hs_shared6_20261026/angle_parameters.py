"""Standalone evaluator for any of the three exported MoTe2 joint constraints."""
import json
from pathlib import Path
import numpy as np

class AngleParameters:
    def __init__(self,directory=None):
        self.directory=Path(directory) if directory is not None else Path(__file__).resolve().parent
        self.record=json.loads((self.directory/'FUNCTION.json').read_text())
        self.mode=self.record['mode'];self.y=np.array(self.record['variables_scaled']);self.sc=np.array(self.record['scales']);self.nodes=np.array(self.record['nodes_deg'])
    @staticmethod
    def basis(theta,j):
        r=np.sin(np.deg2rad(np.atleast_1d(theta))/2)/np.sin(np.deg2rad(3.)/2)
        if j in [0,1]:z=r*r/(1+r*r);return np.c_[np.ones_like(r),z,z*z]
        if j==3:return np.c_[np.ones_like(r),r*r,r**4]
        return np.c_[np.ones_like(r),r,r*r]
    def parameters(self,theta_deg):
        """[a_h/eV A², Vcos/eV, Vsin/eV, w1/eV, V2/eV, w2/eV]."""
        out=np.empty(6);out[0]=self.y[0]*self.sc[0];cursor=1
        for j in range(5):
            constant=self.mode=='all_shared' or (self.mode=='shared_V1_phi_w1' and j<3)
            if constant:out[j+1]=self.y[cursor]*self.sc[j+1];cursor+=1
            else:
                weights=(self.basis(theta_deg,j)@np.linalg.inv(self.basis(self.nodes,j)))[0]
                out[j+1]=weights@self.y[cursor:cursor+3]*self.sc[j+1];cursor+=3
        assert cursor==len(self.y)
        return out
    def physical_parameters(self,theta_deg):
        p=self.parameters(theta_deg)
        return {'mass_over_me':3.80998212/p[0],'V1_meV':1000*np.hypot(p[1],p[2]),'phi_deg':np.degrees(np.arctan2(p[2],p[1])),
                'w1_meV':1000*p[3],'V2_meV':1000*p[4],'w2_meV':1000*p[5]}
