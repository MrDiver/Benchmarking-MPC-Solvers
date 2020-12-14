from MPCBenchmark.agents.agent import Agent
from MPCBenchmark.models.model import Model
import numpy as np


class CEM(Agent):
    """
    max_iter: maximal number of iterations
    n_solution: number of solutions
    num_elites: number of top solution
    epsilon: minimum variance
    alpha: how much of the old mean and variance is used to compute a new mean and variance
    """

    def __init__(self, bounds_low, bounds_high, input_size, output_size, model, params: dict) -> None:
        super().__init__(bounds_low, bounds_high, input_size, output_size, model)
        self.max_iter = params["max_iter"]
        self.n_samples = params["n_samples"]
        self.n_elite = params["n_elite"]
        self.epsilon = params["epsilon"]
        self.alpha = params["alpha"]
        self.instant_cost = params["instant_cost"]
        # Distribution over output parameters
        self.mean = np.zeros(output_size)
        self.variance = params["variance"]

        def f(state, sample):
            # print(sample)
            self.model.predict(state, sample)
            return self.model.get_reward()
        self.f = f

    def calc_action(self, x):
        last_mean = self.mean
        for _ in range(self.max_iter):
            samples = self.mean + \
                np.random.normal(0, np.sqrt(self.variance), size=(
                    self.n_samples, self.output_size))
            print(samples)
            #samples = np.clip(samples, self.bounds_low, self.bounds_high)
            rewards = np.array([self.f(x,sample) for sample in samples])

            # costs = [self.instant_cost(state, 0) for state in states]
            # print("Cost", self.cost_function(x,0))

            elites = samples[np.argsort(-np.array(rewards))][: self.n_elite]
            
            # print("Args:",np.argsort(costs))
            
            new_mean = np.mean(elites, axis=0)
            
            new_var = np.var(elites, axis=0)

            self.mean = self.alpha * self.mean + (1 - self.alpha) * new_mean
            # print(self.mean)
            
            #print(self.mean)
            self.variance = self.alpha * self.variance + (1 - self.alpha) * new_var
            #print(last_mean)
            #print(self.mean)

            if np.abs(last_mean - self.mean) <= self.epsilon:
                break

        return np.clip(self.mean, self.bounds_low, self.bounds_high)
