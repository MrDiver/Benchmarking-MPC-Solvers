import numpy as np

class Model:
    def __init__(self) -> None:
        super().__init__()
        self.last_reward = 0
        self.last_observation = None
    
    def predict(self,current_state:np.ndarray,action:np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def batch_predict(self,states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_reward(self):
        return self.last_reward

    def get_observation(self):
        return self.last_observation