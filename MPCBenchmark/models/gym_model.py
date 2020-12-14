from MPCBenchmark.models.model import Model as Model

import gym
import numpy as np


class GymEnvModel(Model):
    def __init__(self, env_name) -> None:
        super().__init__()
        self.env = gym.make(env_name)
        self.bounds_low = self.env.action_space.low
        self.bounds_high = self.env.action_space.high

    def predict(self, current_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        # if(np.clip(action, self.bounds_low, self.bounds_high) != action):
        #    print("RuntimeWarning: Actions out of action space for this model")
        self.env.reset()
        self.env.env.state = current_state
        obs, r, done, _ = self.env.step(action)
        newstate = self.env.state
        self.last_reward = r
        self.last_observation = obs
        return newstate

    def batch_predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        raise NotImplementedError
