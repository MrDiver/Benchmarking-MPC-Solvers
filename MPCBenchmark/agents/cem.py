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
        self.name="CEM"
        self.K = params["K"]
        self.pred_length = params["T"]
        self.U = np.zeros((self.pred_length, self.output_size), dtype=np.float64)
        self.max_iter = params["max_iter"]
        #self.K = params["n_samples"]
        self.n_elite = params["n_elite"]
        self.epsilon = params["epsilon"]
        self.alpha = params["alpha"]
        self.instant_cost = params["instant_cost"]
        # Distribution over output parameters
        self.std = np.ones((self.pred_length, output_size))*params["std"]
        self.mean = np.zeros((self.pred_length, output_size))
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

    def calc_action(self, x, g_z=None, goal_state=None):
        goal_state = np.array([goal_state])
        if g_z is None:
            if goal_state is None:
                raise AttributeError("goal_state can't be null if no target trajectory g_z is given!")
            g_z = np.repeat(goal_state, self.pred_length,axis=0)
        elif len(np.array(g_z).shape) <= 1:
            raise AttributeError("g_z can't be 1-Dimensional")
        g_z = np.array(g_z)

        #algorithm
        std = self.std.copy()
        for _ in range(self.max_iter):
            samples = np.random.normal(
                self.mean, std, (self.K, self.pred_length, self.output_size))
            samples = np.clip(samples, self.bounds_low, self.bounds_high)
            inputs_ = [(self.model, x, sample) for sample in samples]
            #print(inputs_)
            
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


        xs,_ = self.simulate_trajectory(x,self.U,g_z)
        self.log_iteration(xs, self.mean)
        return np.clip(u0, self.bounds_low, self.bounds_high)

    # TODO: Needs some refactoring
    def simulate_trajectory(self,x,us,g_z):
        xs = np.zeros((self.pred_length+1,self.state_size))
        xs[0,:] = x
        #Simulation
        cost = 0
        for i in range(1,self.pred_length+1):
            newstate = self.model.predict(xs[i-1,:], us[i-1, :], goal=g_z[i-1, :])
            cost += self.model.get_reward()
            xs[i, :] = newstate
        #Simulateend
        return xs, -cost