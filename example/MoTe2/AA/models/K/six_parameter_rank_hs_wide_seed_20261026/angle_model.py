"""Six-term electron model. Exact calibration geometries; interpolated intermediate geometry."""
import json
from pathlib import Path
import numpy as np
from angle_parameters import AngleParameters
from model import extended_response
class AngleModel:
    def __init__(self,directory=None):
        self.directory=Path(directory) if directory else Path(__file__).resolve().parent
        self.law=AngleParameters(self.directory)
        self.rows=json.loads((self.directory/"GEOMETRY.json").read_text())
    def parameters(self,theta):return self.law.parameters(theta)
    def physical_parameters(self,theta):return self.law.physical_parameters(theta)
    def geometry(self,theta,nq=169):
        if theta<=0:raise ValueError("Positive angle required")
        for r in self.rows:
            if abs(theta-r["angle_deg"])<1e-7:
                with np.load(self.directory/"cases"/r["case"]/"model.npz") as z:
                    return {key:z[key].copy() for key in ["q1","q2","reciprocal","kappas","fractional_to_model"]}
        angles=np.array([r["angle_deg"] for r in self.rows]);scales=np.sin(np.deg2rad(angles)/2)
        target=np.sin(np.deg2rad(theta)/2);g={}
        for key in ["reciprocal","kappas","fractional_to_model"]:
            arr=np.array([r[key] for r in self.rows])/scales[:,None,None]
            g[key]=np.array([[np.interp(theta,angles,arr[:,i,j]) for j in range(2)] for i in range(2)])*target
        ij=np.array([(i,j) for i in range(-35,36) for j in range(-35,36)])
        for index,name in enumerate(["q1","q2"]):
            candidates=g["kappas"][index]-ij@g["reciprocal"];norm=np.linalg.norm(candidates,axis=1)
            radius=np.sort(norm)[nq-1];g[name]=candidates[norm<=radius+1e-10]
        return g
    def hamiltonian(self,theta,k,valley="K",nq=169):
        if valley not in ["K","Kprime"]:raise ValueError(valley)
        g=self.geometry(theta,nq);k=np.asarray(k) if valley=="K" else -np.asarray(k)
        h=np.einsum("pij,p->ij",extended_response(k,g["q1"],g["q2"],g["reciprocal"],g["kappas"],"second_harmonic")[0],self.parameters(theta))
        return h if valley=="K" else h.conj()
