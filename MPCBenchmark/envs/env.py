from gym.utils import seeding
import pandas as pd
import numpy as np
from MPCBenchmark.models.model import Model


class Environment:
    def __init__(self, name=None) -> None:
        self.name = "BaseEnvironment" if name is None else name
        self.history = None
        self.state = None
        self.observation = None
        self.model: Model = None
        self.history = pd.DataFrame()

    def __str__(self) -> str:
        return "Name: " + self.name + "\n State: " + self.state

    def seed(self, seed=None):
        self.model.seed(seed)

    def step(self, u):
        self.last_u = u  # for rendering
        self.state = self.model.predict(self.state, u)
        costs = -self.model.get_reward()
        self.history = self.history.append(
            {"state": self.state, "action": u, "cost": costs}, ignore_index=True)
        return self.state, -costs, self._done(), {}

    def reset(self, state=None):
        self.state = self.model.np_random.uniform(
            low=-np.ones(self.model.state_size)*self.model.bounds_low, high=np.ones(self.model.state_size)*self.model.bounds_high)
        if state is not None:
            self.state = state.copy()
        self.last_u = None
        self.history = pd.DataFrame()
        return self.state

    def render(self):
        raise NotImplementedError

    def _done(self):
        raise NotImplementedError
