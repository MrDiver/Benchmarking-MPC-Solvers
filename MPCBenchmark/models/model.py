import numpy as np
from gym.utils import seeding


class Model:
    def __init__(self) -> None:
        super().__init__()
        self.last_reward = 0
        self.last_observation = None
        self.bounds_low = 0
        self.bounds_high = 0
        self.state_size = -1
        self.action_size = -1
        self.np_random, self._seed = seeding.np_random()

    def predict(self, current_state: np.ndarray, action: np.ndarray, goal=None) -> np.ndarray:
        current_state = current_state.reshape(1, -1)
        action = action.reshape(1, -1)
        # print("cur",current_state.shape)
        # print("act",action.shape)
        z = self._transform(current_state, action)
        if goal is None:
            goal = np.zeros(z.shape)
        costs = self._state_cost(z, goal)
        #print("current_state", current_state)
        newstate = self._dynamics(current_state, action)
        #print("newstate", newstate)
        self.last_reward = -costs[0]
        self.last_observation = newstate[0]
        self.last_u = action[0]
        return newstate[0]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def batch_predict(self, current_state: np.ndarray, action: np.ndarray, goal=None) -> np.ndarray:
        z = self._transform(current_state, action)
        if goal is None:
            goal = np.zeros(z.shape)
        costs = self._state_cost(z, goal)
        newstate = self._dynamics(current_state, action)

        self.last_reward = -costs
        self.last_observation = newstate
        self.last_u = action
        return newstate

    def get_reward(self):
        return self.last_reward

    def get_observation(self):
        return self.last_observation

    # used for linearization algorithms to have access to the dynamics

    def _dynamics(self, x, u):
        raise NotImplementedError

    def _transform(self, x, u):
        raise NotImplementedError

    def _state_cost(self, z, g_z):
        raise NotImplementedError

    def _terminal_cost(self, x, g_x):
        raise NotImplementedError


class DummyModel(Model):
    def __init__(self, state_size, action_size) -> None:
        super().__init__()
        self.last_reward = np.array([0])[0]
        self.last_observation = np.zeros_like(state_size)
        self.bounds_low = -1
        self.bounds_high = 1
        self.state_size = state_size
        self.action_size = action_size
        self.W = np.eye(state_size+action_size)

    def predict(self, current_state: np.ndarray, action: np.ndarray, goal=None) -> np.ndarray:
        return current_state

    def batch_predict(self, current_state: np.ndarray, action: np.ndarray, goal=None) -> np.ndarray:
        return current_state
       # used for linearization algorithms to have access to the dynamics

    def _dynamics(self, x, u):
        return x

    def _transform(self, x, u):
        return np.append(x, u, axis=1)

    def _state_cost(self, z, g_z):
        _zd = z-g_z
        #costs = [(z @ self.W) @ z.T for z in _zd]
        costs = np.einsum("bi,ij,bj->b", _zd, self.W, _zd)
        return costs

    def _terminal_cost(self, x, g_x):
        _zd = x-g_x

        costs = np.einsum("bi,ij,bj->b", _zd, self.W, _zd)
        return costs
