from MPCBenchmark.agents.agent import Agent
from MPCBenchmark.models.model import Model
import numpy as np


class MPPI(Agent):
    def __init__(self, bounds_low: np.ndarray, bounds_high: np.ndarray, input_size, output_size, model: Model, params: dict) -> None:
        super().__init__(bounds_low, bounds_high, input_size, output_size, model)
        self.name ="MPPI"
        self.K = params["K"]
        self.pred_length = params["T"]
        self.U = np.zeros((self.pred_length, self.output_size), dtype=np.float64)
        self.std = np.ones((1, 1))*params["std"]
        self.terminal_cost = params["terminal_cost"]
        self.instant_cost = params["instant_cost"]
        self.lam = params["lam"]
        self.x = np.zeros((self.pred_length+1, input_size))

        self.delta_u = np.random.normal(
            0, self.std, (self.K, self.pred_length, self.output_size))
        self.S = np.zeros(self.K)

    def calc_action(self, state, g_z=None, goal_state=None):
        goal_state = np.array([goal_state])
        if g_z is None:
            if goal_state is None:
                raise AttributeError("goal_state can't be null if no target trajectory g_z is given!")
            g_z = np.repeat(goal_state, self.pred_length,axis=0)
        elif len(np.array(g_z).shape) <= 1:
            raise AttributeError("g_z can't be 1-Dimensional")
        g_z = np.array(g_z)

        #algorithm
        for k in range(self.K):
            current_state = state
            for t in range(self.pred_length):
                test_u = self.U[t]+self.delta_u[k, t]
                test_u = np.clip(test_u, self.bounds_low, self.bounds_high)
                current_state = self.model.predict(current_state, test_u,goal=g_z)
                reward = self.model.get_reward()
                cost = self.instant_cost(self.model.get_reward(), test_u)
                cost += -reward
                self.S[k] += cost + self.lam * \
                    test_u.T@np.linalg.pinv(self.std**2)@(self.delta_u[k, t])
                # print(costs)
                # print(self.x[t+1,:])
            self.S[k] += self.terminal_cost(self.model.get_observation())

        beta = self.S.min()
        tmp = np.exp(-(1/self.lam) * (self.S - beta))
        eta = np.sum(tmp)
        w = (1/eta)*tmp

        erg = np.zeros((self.pred_length, self.output_size))
        for t in range(self.pred_length):
            p1 = w*self.delta_u.T[:, t]
            erg[t] = np.sum(p1)

        self.U += erg
        self.U = np.clip(self.U, self.bounds_low,  self.bounds_high)

        u0 = self.U[0]
        #just for logging
        xs,_ = self.simulate_trajectory(state,self.U,g_z)
        self.log_iteration(xs,self.U)
        #end
        self.U = np.roll(self.U, -1)
        self.S[:] = 0  # Resetting cost
        self.U[-1] = 0  # Initializing new action

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