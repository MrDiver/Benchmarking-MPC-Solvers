class Agent:

    def __init__(self, bounds_low, bounds_high, input_size, output_size, model) -> None:
        super().__init__()
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        self.model = model
        self.state_size = input_size
        self.output_size = output_size
        self.name = "BaseAgent"
        self.planning_state_history = []
        self.planning_action_history = []
        self.step_iteration_variable = 0

    def log_iteration(self,planned_x,planned_u):
        #print(self.name,"is adding",planned_x)
        self.planning_state_history.append((self.step_iteration_variable,planned_x))
        self.planning_action_history.append((self.step_iteration_variable,planned_u))
        self.step_iteration_variable+=1

    def reset(self):
        self.planning_state_history = []
        self.planning_action_history = []
        self.step_iteration_variable = 0

    def calc_action(self, state):
        raise NotImplementedError
