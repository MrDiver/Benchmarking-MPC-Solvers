from MPCBenchmark.envs.env import Environment
import gym
import numpy as np
import pandas as pd


class GymWrapperEnv(Environment):
    def __init__(self, gym_name: str, seed=None) -> None:
        super().__init__()
        self.env = gym.make(gym_name)

        self._seed(seed)
        self.bounds_low = self.env.action_space.low
        self.bounds_high = self.env.action_space.high
        obs = self.env.reset()
        self.state = self.env.state
        self.name = "GymWrapper( "+gym_name+", seed = " + str(self.seed) + ")"
        self.history = pd.DataFrame([[self.state, obs, 0, 0]], columns=[
                                    "state", "observation", "action", "reward"])

    def step(self, action):
        if(np.clip(action, self.bounds_low, self.bounds_high) != action):
            raise RuntimeWarning(
                "Actions out of action space for this environmment")
        obs, r, done, extra = self.env.step(action)
        self.state = self.env.state
        self.history = self.history.append(
            {"state": self.state, "observation": obs, "action": action, "reward": r}, ignore_index=True)
        return obs, r, done, extra

    def render(self):
        self.env.render()

    def _seed(self, seed):
        if seed is not None:
            self.seed = seed
        else:
            self.seed = np.random.randint(low=0, high=1000)

        self.env.seed(self.seed)

    def reset(self, seed=None):
        self._seed(seed)
        self.env.reset()
