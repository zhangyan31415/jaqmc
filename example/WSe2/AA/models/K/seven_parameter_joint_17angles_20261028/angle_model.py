import math
import numpy as np
from angle_model_base import AngleModel as Base
class AngleModel(Base):
    def parameters(self,theta):
        lo,hi=self.config['function_domain_deg'];u=(self.coordinate(theta)-self.coordinate(lo))/(self.coordinate(hi)-self.coordinate(lo));d=self.config['degree']
        b=np.array([math.comb(d,i)*u**i*(1-u)**(d-i) for i in range(d+1)])
        lam=np.array(self.config['lambda_controls_eV_A'])@np.array([1-u,u])
        return np.r_[self.config['a_h_eV_A2'],np.array(self.config['potential_controls_eV'])@b,lam]
