class Environment:
    def __init__(self) -> None:
        self.name = "BaseEnvironment"
        self.history = None
        self.state = None
        self.observation = None
        self.bounds_low = [-1]
        self.bounds_high = [1]

    def __str__(self) -> str:
        return "Name: " + self.name + "\n State: " + self.state

    def step(self, action):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def _seed(self, seed):
        raise NotImplementedError

    def reset(self, seed=None):
        raise NotImplementedError
        
