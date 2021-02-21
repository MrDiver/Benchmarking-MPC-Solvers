import numpy as np


class Model:
    def __init__(self) -> None:
        super().__init__()
        self.last_reward = 0
        self.last_observation = None
        self.bounds_low = 0
        self.boudns_high = 0
        self.state_size = -1
        self.action_size = -1

    def predict(self, current_state: np.ndarray, action: np.ndarray, goal_trajectory: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def batch_predict(self, states: np.ndarray, actions: np.ndarray, goal_trajectory: np.ndarray) -> np.ndarray:
        raise NotImplementedError

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
