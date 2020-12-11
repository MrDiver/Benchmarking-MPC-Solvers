import MPCBenchmark.envs.env as Env
import gym
import numpy as np
import pandas as pd

class GymWrapperEnv(Env):
    def __init__(self,gym_name : str,seed=None) -> None:
        super().__init__()
        self.env = gym.make(gym_name)

        if not seed is None:
            self.seed = seed
        else:
            self.seed = np.randint()

        self.env.seed(self.seed)

        obs = self.env.reset()
        self.state = self.env.state
        self.name = "GymWrapper( "+gym_name+", seed = " + self.seed + ")"
        self.history = pd.DataFrame()
        self.history = self.history.append([self.state,obs,0,False])
        
    
    def step(self,action):
        obs,r,done,extra = self.env.step(action)
        self.state = self.env.state
        self.history.append([self.state,obs,r,done])
        return obs,r,done,extra
    

    def render(self):
        self.env.render()

    def reset(self):
        self.env.reset()        
        
