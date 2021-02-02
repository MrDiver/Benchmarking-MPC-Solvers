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

    def __init__(self, *args, **kwargs):#), bounds_low, bounds_high, input_size, output_size, model, params: dict, cores = 4) -> None:
        print(args)
        print(kwargs)
        (bounds_low, bounds_high, input_size, output_size, model, params) = args
        super().__init__(bounds_low, bounds_high, input_size, output_size, model)
        self._pickle_args = args
        self._pickle_kwargs = kwargs
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
        self.pool = mp.Pool(4)

    def __getstate__(self):
        return {"_pickle_args" : self._pickle_args, "_pickle_kwargs": self._pickle_kwargs}

    def __setstate__(self, d):
        out = type(self)(*d["_pickle_args"], **d["_pickle_kwargs"])
        self.__dict__.update(out.__dict__)

    def f(self, state, sample):
        reward = 0
        for at in sample:
            state = self.model.predict(state, at)
            reward += self.model.get_reward()
        return reward

    def f_wrapper(self,x):
        return self.f(*x)

    def calc_action(self, x):
        std = self.std
        for _ in range(self.max_iter):
            samples = np.random.normal(
                self.mean, std, (self.K, self.T, self.output_size))
            samples = np.clip(samples, self.bounds_low, self.bounds_high)
            inputs_ = [(x,sample) for sample in samples]
            rewards = self.pool.map(self.f_wrapper,inputs_)
            #rewards = np.array([self.f_wrapper(x) for x in input_])
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
