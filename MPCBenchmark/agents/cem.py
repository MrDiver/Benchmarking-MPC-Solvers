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

    def __init__(self, bounds_low, bounds_high, input_size, output_size, model, params: dict, cores = 8) -> None:
        super().__init__(bounds_low, bounds_high, input_size, output_size, model)
        self.K = params["K"]
        self.T = params["T"]
        self.U = np.zeros((self.T, self.output_size), dtype=np.float64)
        self.max_iter = params["max_iter"]
        #self.K = params["n_samples"]
        self.n_elite = params["n_elite"]
        self.epsilon = params["epsilon"]
        self.alpha = params["alpha"]
        self.instant_cost = params["instant_cost"]
        # Distribution over output parameters
        self.std = np.ones((self.T, output_size))*params["std"]
        self.mean = np.zeros((self.T, output_size))
        self.pool = mp.Pool(cores)

    @staticmethod
    def f(model, state, sample):
        reward = 0
        for at in sample:
            state = model.predict(state, at)
            reward += model.get_reward()
        return reward

    @staticmethod
    def f_wrapper(x):
        return CEM.f(*x)

    def calc_action(self, x):
        std = self.std
        def create_inputs(sample):
            return (self.model, x, sample)

        for _ in range(self.max_iter):
            samples = np.random.normal(
                self.mean, std, (self.K, self.T, self.output_size))
            samples = np.clip(samples, self.bounds_low, self.bounds_high)
            
            inputs_ = [(self.model, x, sample) for sample in samples]
            rewards = np.array(self.pool.map(CEM.f_wrapper,inputs_))
            #rewards = np.array([self.f_wrapper(x) for x in inputs_])
            elites = samples[np.argsort(-rewards)][: self.n_elite]

            new_mean = np.mean(elites, axis=0)
            new_std = np.std(elites, axis=0)
            self.mean = self.alpha * self.mean + (1 - self.alpha) * new_mean
            std = self.alpha * std + (1 - self.alpha) * new_std

            u0 = self.mean[0]
            self.mean = np.roll(self.mean, -1)
            self.mean[-1] = 0
            if (std < self.epsilon).all():
                break

        return np.clip(u0, self.bounds_low, self.bounds_high)
