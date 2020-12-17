from MPCBenchmark.agents.agent import Agent
from MPCBenchmark.models.model import Model
import numpy as np


class CEM(Agent):
    """
    max_iter: maximal number of iterations
    n_samples: number of samples
    n_elites: number of top solution
    epsilon: minimum variance
    alpha: how much of the old mean and variance is used to compute a new mean and variance
    """

    def __init__(self, bounds_low, bounds_high, input_size, output_size, model, params: dict) -> None:
        super().__init__(bounds_low, bounds_high, input_size, output_size, model)
        self.K = params["K"]
        self.T = params["T"]
        self.U = np.zeros((self.T, self.output_size), dtype=np.float64)
        self.max_iter = params["max_iter"]
        self.n_samples = params["n_samples"]
        self.n_elite = params["n_elite"]
        self.epsilon = params["epsilon"]
        self.alpha = params["alpha"]
        self.instant_cost = params["instant_cost"]
        # Distribution over output parameters
        self.variance = params["variance"]
        self.mean = np.zeros((self.T, self.output_size))

        def f(state, sample):
            reward = 0
            for at in sample:
                state = self.model.predict(state, at)
                reward += self.model.get_reward()
            return reward
        self.f = f

    def calc_action(self, x):
        variance = self.variance
        for _ in range(self.max_iter):
            samples = self.mean + np.random.normal(0, np.sqrt(variance), size=(
                (self.n_samples, self.T, self.output_size)))

            # print(samples)
            samples = np.clip(samples, self.bounds_low, self.bounds_high)

            rewards = np.array([self.f(x, sample) for sample in samples])
            # print("samples",samples)
            # costs = [self.instant_cost(state, 0) for state in states]
            # print("Cost", self.cost_function(x,0))

            elites = samples[np.argsort(-rewards)][: self.n_elite]
            # print(-rewards)
            # print(samples[np.argsort(-rewards)].flatten())
            # return 0
            # print("elites",elites)
            # print("Args:",np.argsort(costs))

            new_mean = np.mean(elites, axis=0)
            # print("Mean",new_mean)
            new_var = np.var(elites, axis=0)
            new_var = np.mean(new_var)
            # print(new_var)

            self.mean = self.alpha * self.mean + (1 - self.alpha) * new_mean
            # print(self.mean)

            # print(self.mean)
            variance = self.alpha * variance + (1 - self.alpha) * new_var
            # print(variance)
            # print(self.mean)
            u0 = self.mean[0]
            self.mean = np.roll(self.mean, -1)
            self.mean[-1] = 0
            if variance < self.epsilon:
                break

        return np.clip(u0, self.bounds_low, self.bounds_high)
