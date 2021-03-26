from gym.utils import seeding
import pandas as pd
import numpy as np
from MPCBenchmark.models.model import Model


class Environment:
    def __init__(self, name=None) -> None:
        self.name = "BaseEnvironment" if name is None else name
        self.history = None
        self.true_state = None
        self.state = None
        self.observation = None
        self.model: Model = None
        self.history = pd.DataFrame()
        self.actuation_noise = False
        self.sensor_noise = False
        self.actuation_noise_std = 0
        self.sensor_noise_std = 0

    def __str__(self) -> str:
        return "Name: " + self.name + "\n State: " + self.true_state

    def set_actuation_noise(self, actuation_std):
        self.actuation_noise = True
        self.actuation_noise_std = actuation_std

    def set_sensor_noise(self, sensor_std):
        self.sensor_noise = True
        self.sensor_noise_std = sensor_std

    def seed(self, seed=None):
        self.model.seed(seed)

    def step(self, u):
        self.last_u = true_u = u
        print("u", u)
        if self.actuation_noise:
            u += np.random.normal(0, self.actuation_noise_std,
                                  self.model.action_size)
        print("u_noise", u)
        # for rendering
        self.true_state = self.model.predict(self.true_state, u)
        self.state = self.true_state + \
            (np.random.normal(
                0, self.sensor_noise_std, self.model.state_size) if self.sensor_noise else 0)
        costs = -self.model.get_reward()
        self.history = self.history.append(
            {"state": self.state, "true_state": self.true_state, "action": u, "true_action": true_u, "cost": costs}, ignore_index=True)
        return self.true_state, -costs, self._done(), {}

    def reset(self, state=None):
        self.true_state = self.state = self.model.np_random.uniform(
            low=-np.ones(self.model.state_size)*self.model.bounds_low, high=np.ones(self.model.state_size)*self.model.bounds_high)
        if state is not None:
            self.true_state = state.copy()
        self.last_u = None
        self.history = pd.DataFrame()
        return self.true_state

    def render(self):
        raise NotImplementedError

    def _done(self):
        raise NotImplementedError
