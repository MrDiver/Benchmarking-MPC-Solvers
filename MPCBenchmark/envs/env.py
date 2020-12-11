class Env:
    def __init__(self) -> None:
        super().__init__()
        self.name = "BaseEnvironment"
        self.history = None
        self.state = None
        self.observation = None
    

    def __str__(self) -> str:
        return "Name: " + self.name + "\n State: " + self.state