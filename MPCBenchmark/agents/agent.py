from MPCBenchmark.models.model import Model
import numpy as np
class Agent:

    def __init__(self, name : str, model : Model, logging : bool = True) -> None:
        super().__init__()
        self.bounds_low = model.bounds_low
        self.bounds_high = model.bounds_high
        self.state_size = model.state_size
        self.action_size = model.action_size
        self.model = model
        self.name = name
        self.logging = logging
        # For plotting and logging
        self.planned_xs = None
        self.planned_us = None
        self.planned_costs = None
        self.planning_state_history = []
        self.planning_action_history = []
        self.planning_costs_history = []
        self.step_iteration_variable = 0

    def predict_action(self, state, g_z=None, goal_state=None):
        #Setting up g_z
        goal_state = np.array([goal_state])
        if g_z is None:
            if goal_state is None:
                raise AttributeError("goal_state can't be null if no target trajectory g_z is given!")
            g_z = np.repeat(goal_state, self.horizon_length,axis=0)
        elif len(np.array(g_z).shape) <= 1:
            raise AttributeError("g_z can't be 1-Dimensional")
        g_z = np.array(g_z)
        #end

        u0 = self._calc_action(state,g_z)
        self.planned_us = np.clip(self.planned_us,self.bounds_low,self.bounds_high)

        if self.logging:
            self.planned_xs,self.planned_costs = self.simulate_trajectory(state,self.planned_us,g_z)
            self._log_iteration(self.planned_xs,self.planned_us,self.planned_costs)
            self.planned_us = np.roll(self.planned_us,-1)
            self.planned_us[-1] = self.planned_us[-2]

        self.step_iteration_variable+=1
        return np.clip(u0, self.bounds_low, self.bounds_high)

    def _calc_action(self,x, g_z=None):
        raise NotImplementedError

    def _log_iteration(self,planned_xs,planned_us,planned_costs):
        #print(self.name,"is adding",planned_x)
        self.planning_state_history.append((self.step_iteration_variable,planned_xs.copy()))
        self.planning_action_history.append((self.step_iteration_variable,planned_us.copy()))
        self.planning_costs_history.append(planned_costs.copy())

    def reset(self):
        self.planning_state_history = []
        self.planning_action_history = []
        self.step_iteration_variable = 0

    def _calc_action(self, state):
        raise NotImplementedError

    def simulate_trajectory(self,x,us,g_z):
        xs = np.zeros((self.horizon_length+1,self.state_size))
        xs[0,:] = x
        #Simulation
        cost = 0
        for i in range(1,self.horizon_length+1):
            newstate = self.model.predict(xs[i-1,:], us[i-1, :], goal=g_z[i-1, :])
            cost += self.model.get_reward()
            xs[i, :] = newstate
        #Simulateend
        return xs, -cost