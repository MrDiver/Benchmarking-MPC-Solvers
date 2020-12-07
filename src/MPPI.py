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
    def __init__(self,K,T,ctrl_shape,F=None,U=None,Sigma=None,tc=None,ic=None,l=0.02):
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
        #x_used = np.zeros((self.T,len(x)))
        #x_used[0,:] = x
        S = np.zeros(self.K)
        Epsilon = np.zeros((self.K, self.T))
        #print(self.U)
        # Simulate k sampled movements
        for k in range(self.K):
            #AB HIER QDT UND QD NICHTS ANDERES 
            qt,qdt = q.copy(),qd.copy()
            for t in range(self.T):
                Epsilon[k,t] = np.random.normal(0, self.Sigma).ravel()
                # Epsilon[k,t] = np.random.multivariate_normal(self.U[t].flatten(), self.Sigma, 1).ravel()
            # Simulate the System for T Steps
            for t in range(1,self.T):
                qt,qdt = self.F(qt,qdt, self.U[t-1] + Epsilon[k,t-1])
                # accumulate moving cost
                S[k] += self.ic(qt,qdt,self.U[t-1]) + self.l * self.U[t-1].T * np.linalg.pinv(self.Sigma) * Epsilon[k,t-1]
            # add terminal cost phi = tc
            S[k] += self.tc(qt,qdt)
        # take best sample
        beta = S.min()
        # calculate magic with monte carlo estimate, used in the importance sampling weight
        eta = np.sum(np.exp(-1/self.l * (S-beta)))
        # importance sampling weight
        w = 1/eta * np.exp(-1/self.l*(S-beta))
        #print(w)
        # compute U
        # Eps -> (K x T)
        for t in range(self.T):
            self.U[t] += np.dot(np.hstack(w),np.hstack(Epsilon[:,t]))
        u0 = self.U[0]
        # update U
        self.U[:-1]=self.U[1:]
        return u0

    def print_state(self):
        print(self.U)




# # hyparams
# """
# Sigma: variance of the gaussian
# phi: terminal cost (stronger than state cost)
# q: instantaneous state cost / density function
# l: scaling factor for Jensens Inequality
# """
# def mppi(F,K,T,U,Sigma,phi,q,l, debug=True):
#     # Initialize Env
#     taskNotCompleted = True
    
#     def getInitialState():
#         return 0

#     while taskNotCompleted:
#         x0 = getInitialState()
#         x = np.zeros((T,len(x0)))
#         x[0,:] = x0
#         S = np.zeros(K)
#         # Simulate k sampled movements
#         for k in range(K): 
#             Epsilon = np.random.multivariate_normal(U[k], Sigma, T) # Do we need to compute this ? (Sigma)
#             # Simulate the System for T Steps
#             for t in range(1,T):
#                 x[t,:] = F(x[t-1,:], U[t-1] + Epsilon[t-1])
#                 # accumulate moving cost
#                 S += q(x[t]) + l * U[t-1].T * np.inv(Sigma) * Epsilon[t-1]
#             # add terminal cost
#             S += phi(x[t])
#         # take best sample
#         beta = np.min(S)
#         #calculate magic
#         eta = np.sum(np.exp(-1/l * (S-beta)))

#         #for
#         w = 1/eta * np.exp(-1/l*(S-beta))

#         #for
#         U += np.sum(w*Epsilon,axis=0)
#         #sendtoactuators
#         u0 = U[0]
#         #for
#         U[:-1] = U[1:]
#         #init
#         U[-1,:] = np.zeros(u0.shape)
