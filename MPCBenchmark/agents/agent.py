class Agent:

    def __init__(self, bounds_low, bounds_high, input_size, output_size, model) -> None:
        super().__init__()
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        self.model = model
        self.state_size = input_size
        self.output_size = output_size
        self.name = "BaseAgent"

    def calc_action(self, state):
        raise NotImplementedError
