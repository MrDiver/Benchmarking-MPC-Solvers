from MPCBenchmark.agents.agent import Agent
from MPCBenchmark.models.model import Model
import numpy as np


class MPPI(Agent):
    def __init__(self, model: Model, params: dict) -> None:
        super().__init__("MPPI", model)
        self.K = params["K"]
        self.horizon_length = params["T"]
        self.planned_us = np.zeros((self.horizon_length, self.action_size), dtype=np.float64)
        self.std = np.ones((1, 1))*params["std"]
        self.terminal_cost = params["terminal_cost"]
        self.instant_cost = params["instant_cost"]
        self.lam = params["lam"]
        self.x = np.zeros((self.horizon_length+1, self.state_size))

        self.delta_u = np.random.normal(
            0, self.std, (self.K, self.horizon_length, self.action_size))
        self.sample_costs = np.zeros(self.K)


    #@staticmethod
    #def f(model, state, sample, g_z):
    #    reward = 0
    #    for at in sample:
    #        state = model.predict(state, at)
    #        reward += model.get_reward()
    #    return reward

    #@staticmethod
    #def f_wrapper(x):
    #    return MPPI.f(*x)

    def _calc_action(self, x, g_z):

        for k in range(self.K):
            current_x = x
            for t in range(self.horizon_length):
                test_u = self.planned_us[t]+self.delta_u[k, t]
                test_u = np.clip(test_u, self.bounds_low, self.bounds_high)
                current_x = self.model.predict(current_x, test_u,goal=g_z)
                reward = self.model.get_reward()
                cost = self.instant_cost(self.model.get_reward(), test_u)
                cost += -reward
                self.sample_costs[k] += cost + self.lam * \
                    test_u.T@np.linalg.pinv(self.std**2)@(self.delta_u[k, t])
                # print(costs)
                # print(self.x[t+1,:])
            self.sample_costs[k] += self.terminal_cost(self.model.get_observation())

        beta = self.sample_costs.min()
        tmp = np.exp(-(1/self.lam) * (self.sample_costs - beta))
        eta = np.sum(tmp)
        w = (1/eta)*tmp

        erg = np.zeros((self.horizon_length, self.action_size))
        for t in range(self.horizon_length):
            p1 = w*self.delta_u.T[:, t]
            erg[t] = np.sum(p1)

        self.planned_us += erg

        u0 = self.planned_us[0]

        self.sample_costs[:] = 0  # Resetting cost

        return u0