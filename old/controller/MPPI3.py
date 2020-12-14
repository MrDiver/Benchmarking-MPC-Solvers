from .core import Algorithm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# @jitclass([("K",int32),("T",int32),("output_size",int32),
#           ("l",float64),("input_size",int32),("U",float64[:]),("Sigma",float64[:]),
#           ("x",float64[:]),
#           ("delta_u",float64[:]),("zero_vec",float64[:]),("S",float64[:])])


class MPPI3(Algorithm):
    """
    K: number of samples
    T: numbers of timesteps
    U: initial control Sequence
    Sigma: variance of the gaussian
    tc: terminal cost (stronger than state cost)
    ic: instantaneous state cost / density function
    l: scaling factor for Jensens Inequality
    """

    def __init__(self, K, T, output_size, input_size, F=(lambda x, u: x), U=None, Sigma=None, tc=(lambda x: 0), ic=(lambda x, u: 0), l=1, env=None, env_initial=None):
        super()
        self.K = K
        self.T = T
        self.F = F
        self.U = np.zeros((T, output_size), dtype=np.float64)
        self.Sigma = Sigma
        self.phi = tc
        self.q = ic
        self.l = l
        self.input_size = input_size
        self.output_size = output_size
        self.x = np.zeros((T+1, input_size))
        np.random.seed(0)

        self.delta_u = np.random.normal(
            0, self.Sigma, (self.K, self.T, self.output_size))

        self.zero_vec = np.zeros(self.output_size)
        self.S = np.zeros(K)

        # print("U shape",self.U.shape)
        # print("noise shape",self.delta_u.shape)
        # print("cost shape",self.S.shape)
        # print("noise",self.delta_u)

    def step(self, t, x_init, env_state):
        # self.delta_u = np.random.normal(0,self.Sigma,(self.K,self.T,self.output_size))
        for k in range(self.K):
            self.x[0, :] = x_init.copy()
            current_state = env_state
            for t in range(self.T):
                test_u = self.U[t]+self.delta_u[k, t]
                #print("perturbed action ",t," is ",test_u)

                #test_u = np.clip(test_u,-2,2)
                # self.F(self.x[t,:],test_u)
                current_state, reward, self.x[t+1,
                                              :] = self.F(current_state, test_u)
                cost = self.q(self.x[t+1, :], test_u)
                cost += -reward
                # + self.l*test_u.T@np.linalg.pinv(self.Sigma)@(self.delta_u[k,t])
                self.S[k] += cost
                # print(costs)
                # print(self.x[t+1,:])
            self.S[k] += self.phi(self.x[self.T-1, :])

        beta = self.S.min()
        # mink = self.S.argmin()
        cost_thing = np.exp(-(1/self.l) * (self.S - beta))
        eta = np.sum(cost_thing)
        w = (1/eta)*cost_thing

        # plt.figure()
        # plt.plot(w)
        # plt.show()

        #erg = (self.delta_u.T*w).sum(axis=2).T
        # macht genau das gleiche wie das darüber
        erg = np.zeros((self.T, self.output_size))
        # print("w",w)
        for t in range(self.T):
            p1 = w*self.delta_u.T[:, t]
            erg[t] = np.sum(p1)

        self.U += erg

        # self.U = np.clip(self.U,-2,2)
        u0 = self.U[0].copy()

        # plt.plot(self.U)
        # plt.scatter(mink,1)
        # plt.show()

        self.U = np.roll(self.U, -1)
        self.S[:] = 0
        self.U[-1] = 0  # self.U[-2]

        return u0

    def print_state(self):
        print(self.U)
