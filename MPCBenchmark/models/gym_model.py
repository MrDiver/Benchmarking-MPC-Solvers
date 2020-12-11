from MPCBenchmark.models.model import Model as Model

import gym
import numpy as np

class GymEnvModel(Model):
    def __init__(self, env_name) -> None:
        super().__init__()
        self.env = gym.make("env_name")


    def predict(self,current_state:np.ndarray,action:np.ndarray) -> np.ndarray:
        self.env.env.state = current_state
        obs,r,done,_ = self.e.step(action)
        newstate = self.e.env.state
        self.last_reward = r
        self.last_observation = obs
        return newstate

    
    def batch_predict(self,states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        raise NotImplementedError