from .core import Algorithm
import numpy as np

class MPPI(Algorithm):
    """
    K: number of samples
    T: numbers of timesteps
    U: initial control Sequence
    Sigma: variance of the gaussian
    tc: terminal cost (stronger than state cost)
    ic: instantaneous state cost / density function
    l: scaling factor for Jensens Inequality
    """
    def __init__(self,K,T,ctrl_shape,F=None,U=None,Sigma=None,tc=None,ic=None,l=None):
        super()
        self.K=K
        self.T=T
        self.ctrl_shape = ctrl_shape
        self.F=F
        self.U = np.zeros((T,*ctrl_shape))
        self.Sigma = Sigma
        self.tc = tc
        self.ic = ic
        self.l = l
        

    def step(self,t,q,qd):
        

        u = self.U[0]
        self.U[:-1]=self.U[1:]
        return u

    def print_state(self):
        print(self.U)