from MPCBenchmark.agents.agent import Agent
from MPCBenchmark.models.model import Model
import numpy as np
from multiprocessing import Pool


class MPPI(Agent):
    name = "MPPI"

    def __init__(self, model: Model, params: dict, cores=8) -> None:
        super().__init__("MPPI", model)
        self.K = params["K"]
        self.horizon_length = params["T"]
        self.planned_us = np.zeros(
            (self.horizon_length, self.action_size), dtype=np.float64)
        self.std = np.ones((1, 1))*params["std"]
        self.lam = params["lam"]
        self.x = np.zeros((self.horizon_length+1, self.state_size))

        self.delta_u = np.random.normal(
            0, self.std, (self.K, self.horizon_length, self.action_size))
        self.sample_costs = np.zeros(self.K)

        self.pool = Pool(cores)

    # def __del__(self):
    #    print("Deleting MPPI")
    #    self.pool.close()

    @staticmethod
    def f(model, state, planned_us, delta_us, g_z, horizon_length, lam, std):
        current_x = state
        sample_cost = 0
        sample = planned_us + delta_us
        for t in range(horizon_length):
            test_u = sample[t]  # self.planned_us[t]+self.delta_u[k, t]
            #test_u = np.clip(test_u, self.bounds_low, self.bounds_high)
            current_x = model.predict(current_x, test_u, goal=g_z)
            cost = -model.get_reward()
            sample_cost += cost + lam * \
                test_u.T@np.linalg.pinv(std**2)@(delta_us[t])
        #sample_costs += self.model._terminal_cost(current_x,g_z[:])
        return sample_cost

    @staticmethod
    def f_wrapper(x):
        return MPPI.f(*x)

    def _calc_action(self, x, g_z):
        _inputs = [(self.model, x, self.planned_us, self.delta_u[k], g_z,
                    self.horizon_length, self.lam, self.std) for k in range(self.K)]
        self.sample_costs = np.array(self.pool.map(MPPI.f_wrapper, _inputs))
        #self.sample_costs = np.array([MPPI.f_wrapper(x) for x in _inputs])
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
