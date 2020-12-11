class Environment:
    def __init__(self) -> None:
        self.name = "BaseEnvironment"
        self.history = None
        self.state = None
        self.observation = None
    

    def __str__(self) -> str:
        return "Name: " + self.name + "\n State: " + self.state