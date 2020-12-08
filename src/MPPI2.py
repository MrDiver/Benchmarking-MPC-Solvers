from .core import Algorithm
import numpy as np
import matplotlib.pyplot as plt

class MPPI2(Algorithm):
    """
    K: number of samples
    T: numbers of timesteps
    U: initial control Sequence
    Sigma: variance of the gaussian
    tc: terminal cost (stronger than state cost)
    ic: instantaneous state cost / density function
    l: scaling factor for Jensens Inequality
    """
    def __init__(self,K,T,output_size,input_size,F=None,U=None,Sigma=None,tc=None,ic=None,l=0.02):
        super()
        self.K=K
        self.T=T
        self.ctrl_shape = output_size
        self.F=F
        self.U = np.zeros((T,output_size))
        self.Sigma = Sigma
        self.phi = tc
        self.q = ic
        self.l = l
        self.input_size = input_size
        self.output_size = output_size
        
        self.x = np.zeros((T,input_size))
        self.delta_u = None # np.zeros((K,T,output_size))
        
        self.zero_vec = np.zeros(self.output_size)
        self.S = np.zeros(K)
        
    def step(self,t,x_init):
        # Das geht so viel viel viel viel viel viel viel schneller um genau zu sein 90mal schneller als vorher
        self.delta_u = np.random.multivariate_normal(self.zero_vec,self.Sigma,size=(self.K,self.T))
        #print(self.delta_u.shape)
        for k in range(self.K):
            self.x[0,:] = x_init
            for t in range(self.T-1):
                self.x[t+1,:] = self.F(self.x[t,:],self.U[t]+self.delta_u[k,t])
                self.S[k] += self.q(self.x[t+1,:],self.U[t]) + self.l*self.U[t].T@np.linalg.pinv(self.Sigma)@(self.delta_u[k,t])
                #print(self.x[t+1,:])
            self.S[k] += self.phi(self.x[self.T-1,:])
            
        beta = self.S.min()
        mink = self.S.argmin()
        eta = np.sum(np.exp(-1/self.l*(self.S-beta)))
        w = 1/eta*np.exp(-1/self.l*(self.S-beta))
        erg = (self.delta_u.T*w).sum(axis=2).T
        #macht genau das gleiche wie das darüber
        # erg2 = np.zeros(self.T)
        # for t in range(self.T):
        #     p1 = w*self.delta_u.T[0,t]
        #     erg2[t] = np.sum(p1)

        self.U += erg
        
        u0 = self.U[0] + self.delta_u[mink,0]#self.U[0]
        
        
        # plt.plot(self.U)
        # plt.scatter(mink,1)
        # plt.show()
        
        self.U[:-1] = self.U[1:]
        self.U[-1] = 0
        
        return u0
        
        

    def print_state(self):
        print(self.U)
