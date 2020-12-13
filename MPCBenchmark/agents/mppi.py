from MPCBenchmark.agents.agent import Agent
from MPCBenchmark.models.model import Model
import numpy as np


class MPPI(Agent):
    def __init__(self, bounds_low: np.ndarray, bounds_high: np.ndarray, input_size, output_size, model: Model, params: dict) -> None:
        super().__init__(bounds_low, bounds_high, input_size, output_size, model)
        self.K = params["K"]
        self.T = params["T"]
        self.U = np.zeros((self.T, self.output_size), dtype=np.float64)
        self.Sigma = params["Sigma"]
        self.terminal_cost = params["terminal_cost"]
        self.instant_cost = params["instant_cost"]
        self.lam = params["lam"]
        self.x = np.zeros((self.T+1, input_size))

        self.delta_u = np.random.normal(
            0, self.Sigma, (self.K, self.T, self.output_size))
        self.S = np.zeros(self.K)

    def calc_action(self, state):
        for k in range(self.K):
            current_state = state
            for t in range(self.T):
                test_u = self.U[t]+self.delta_u[k, t]
                test_u = np.clip(test_u, self.bounds_low, self.bounds_high)
                current_state = self.model.predict(current_state, test_u)
                reward = self.model.get_reward()
                cost = self.instant_cost(self.model.get_reward(), test_u)
                cost += -reward
                self.S[k] += cost + self.lam * \
                    test_u.T@np.linalg.pinv(self.Sigma)@(self.delta_u[k, t])
                # print(costs)
                # print(self.x[t+1,:])
            self.S[k] += self.terminal_cost(self.model.get_observation())

        beta = self.S.min()
        tmp = np.exp(-(1/self.lam) * (self.S - beta))
        eta = np.sum(tmp)
        w = (1/eta)*tmp

        erg = np.zeros((self.T, self.output_size))
        for t in range(self.T):
            p1 = w*self.delta_u.T[:, t]
            erg[t] = np.sum(p1)

        self.U += erg
        self.U = np.clip(self.U, self.bounds_low,  self.bounds_high)

        u0 = self.U[0]

        self.U = np.roll(self.U, -1)
        self.S[:] = 0  # Resetting cost
        self.U[-1] = 0  # Initializing new action

        return np.clip(u0, self.bounds_low, self.bounds_high)
