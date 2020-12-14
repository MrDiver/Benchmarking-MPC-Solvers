from .core import Algorithm
import numpy as np

class CEM(Algorithm):
    """
    max_iter: maximal number of iterations
    n_solution: number of solutions
    num_elites: number of top solution
    epsilon: minimum variance
    alpha: how much of the old mean and variance is used to compute a new mean and variance
    """
    def __init__(self, input_size, output_size, max_iter, n_samples, num_elite, cost_function=None, F=None, upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25):
        super()
        self.input_size, self.output_size, self.max_iter, self.n_samples, self.num_elite = input_size, output_size, max_iter, n_samples, num_elite
        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha
        self.cost_function = cost_function
        self.F = F
        # Distribution over output parameters
        self.mean = np.zeros(output_size)
        self.variance = np.eye(output_size)*3

    def step(self, t, x):
        last_mean = self.mean
        for _ in range(self.max_iter):
            samples = self.mean + np.random.normal(0, self.variance, size=(self.n_samples,self.output_size))
            states = [self.F(x, sample) for sample in samples]
            costs = [self.cost_function(state, 0) for state in states]
            #print("Cost", self.cost_function(x,0))
            elites = samples[np.argsort(costs)][:self.num_elite]
            #print("Samples:", costs)
            #print("Args:",np.argsort(costs))
            new_mean = np.mean(elites, axis=0)
           # new_var = np.var(elites, axis=0)

            self.mean = self.alpha * self.mean + (1 - self.alpha) * new_mean
            #self.variance = self.alpha * self.variance + (1 - self.alpha) * new_var

            if np.abs(last_mean - self.mean) <= self.epsilon:
                break

        sol, solvar = self.mean, self.variance
        return sol
