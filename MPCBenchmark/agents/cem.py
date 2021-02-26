from MPCBenchmark.agents.agent import Agent
from MPCBenchmark.models.model import Model
import numpy as np
from gym.utils import EzPickle
import multiprocessing as mp


class CEM(Agent):
    """
    max_iter: maximal number of iterations
    n_samples: number of samples
    n_elites: number of top solution
    epsilon: minimum variance
    alpha: how much of the old mean and variance is used to compute a new mean and variance
    """

    def __init__(self, model: Model, params: dict, cores: int = 8) -> None:
        super().__init__("CEM", model)
        self.K = params["K"]
        self.horizon_length = params["T"]
        self.max_iter = params["max_iter"]
        #self.K = params["n_samples"]
        self.n_elite = params["n_elite"]
        self.epsilon = params["epsilon"]
        self.alpha = params["alpha"]
        self.instant_cost = params["instant_cost"]
        # Distribution over output parameters
        self.std = np.ones(
            (self.horizon_length, self.action_size))*params["std"]
        self.planned_us = np.zeros((self.horizon_length, self.action_size))
        self.pool = mp.Pool(cores)

    @staticmethod
    def f(model, state, sample, g_z):
        reward = 0
        for u_t in sample:
            state = model.predict(state, u_t, g_z)
            reward += model.get_reward()
        return reward

    @staticmethod
    def f_wrapper(x):
        return CEM.f(*x)

    def _calc_action(self, x, g_z):

        std = self.std.copy()
        for _ in range(self.max_iter):
            samples = np.random.normal(
                self.planned_us, std, (self.K, self.horizon_length, self.action_size))
            samples = np.clip(samples, self.bounds_low, self.bounds_high)
            inputs_ = [(self.model, x, sample, g_z) for sample in samples]
            # print(inputs_)

            rewards = np.array(self.pool.map(CEM.f_wrapper, inputs_))
            #rewards = np.array([self.f_wrapper(x) for x in inputs_])

            elites = samples[np.argsort(-rewards)][: self.n_elite]

            new_mean = np.mean(elites, axis=0)
            new_std = np.std(elites, axis=0)
            self.planned_us = self.alpha * \
                self.planned_us + (1 - self.alpha) * new_mean
            std = self.alpha * std + (1 - self.alpha) * new_std

            if (std < self.epsilon).all():
                break
        u0 = self.planned_us[0]
        return u0
