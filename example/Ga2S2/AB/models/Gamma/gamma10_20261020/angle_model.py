"""Portable ten-parameter Gamma model. Requires NumPy and four local files.
Default H uses fixed integer shell5 (Q91) for every angle. source_hamiltonian
uses exact stored source Q at known angles for reference-state comparisons.
Raw matrices use eV, momenta Angstrom^-1, input fractional moire k.
Only dispersion relative to each model/reference's own Gamma minimum is
calibrated. Subtract that origin once per entire path, never separately per k.
"""
from pathlib import Path
import importlib.util,json
import numpy as np
HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_self_contained_gamma_physics',HERE/'physical.py')
_physics=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_physics)
physical_parameter_names=list(_physics.PHYSICAL_NAMES)
global_coefficient_names=['a_eV_A2','b_eV_inverse','eta','lambda_A2','epsilon']+[name+'_'+mode for name in ['Vc_eV','Vs_eV','t0_eV','tc_eV','ts_eV'] for mode in ['A','B']]

class AngleModel:
 def __init__(self,coefficients=None,geometry=None):
  self.coefficients=np.asarray(coefficients if coefficients is not None else json.loads((HERE/'COEFFICIENTS.json').read_text())['endpoint_controls'],float)
  self.geometry=geometry if geometry is not None else np.load(HERE/'geometry.npz');self.angles=np.asarray(self.geometry['angles']);self.labels=np.asarray(self.geometry['labels']);self.physical_parameter_names=physical_parameter_names;self.global_coefficient_names=global_coefficient_names
 def _known(self,theta):
  distance=np.minimum(abs(self.angles-float(theta)),abs(self.labels-float(theta)));i=int(distance.argmin());return i if distance[i]<1e-7 else None
 def _canonical_theta(self,theta):
  i=self._known(theta);return float(theta) if i is None else float(self.angles[i])
 def parameters(self,theta_deg):
  p,j=_physics.parameters(self.coefficients,self._canonical_theta(theta_deg));return dict(zip(physical_parameter_names,map(float,p)))
 def reciprocal(self,theta_deg):
  i=self._known(theta_deg)
  if i is not None:return np.asarray(self.geometry[f'b_{i}'])
  # Geometry-only interpolation; no stored Hamiltonians, bands, or eigenstates.
  angles=self.angles;r=np.sin(np.deg2rad(angles)/2);target=np.sin(np.deg2rad(float(theta_deg))/2);scaled=np.array([self.geometry[f'b_{j}']/r[j] for j in range(len(angles))]);return np.array([[np.interp(target*target,r*r,scaled[:,a,b]) for b in range(2)] for a in range(2)])*target
 def q(self,theta_deg,radius=5):return _physics.integer_shell(int(radius))@self.reciprocal(theta_deg)
 def hamiltonian(self,theta_deg,k_fractional,radius=5):
  p=np.array(list(self.parameters(theta_deg).values()));b=self.reciprocal(theta_deg);return _physics.full(self.q(theta_deg,radius),b,np.asarray(k_fractional,float),p)
 def source_hamiltonian(self,theta_deg,k_fractional):
  i=self._known(theta_deg)
  if i is None:raise ValueError('Source Q exists only at exact configured angles or their recorded rounded labels')
  p=np.array(list(self.parameters(theta_deg).values()));return _physics.full(self.geometry[f'q_{i}'],self.geometry[f'b_{i}'],np.asarray(k_fractional,float),p)
_MODEL=None
def _get():
 global _MODEL
 if _MODEL is None:_MODEL=AngleModel()
 return _MODEL
def hamiltonian(theta_deg,k_fractional):return _get().hamiltonian(theta_deg,k_fractional)
def source_hamiltonian(theta_deg,k_fractional):return _get().source_hamiltonian(theta_deg,k_fractional)
def parameters(theta_deg):return _get().parameters(theta_deg)
