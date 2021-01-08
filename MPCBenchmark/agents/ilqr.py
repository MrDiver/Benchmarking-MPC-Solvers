import numpy as np
from MPCBenchmark.agents.agent import Agent

class ILQR(Agent):

    def __init__(self, bounds_low, bounds_high, input_size, output_size, model, params: dict) -> None:
        super().__init__(bounds_low, bounds_high, input_size, output_size, model)

        self.max_iter = params["max_iter"]
        self.mu = params["mu"]
        self.max_mu = params["max_mu"]
        self.min_mu = params["min_mu"]
        self.init_mu = params["init_mu"]
        self.init_delta = params["init_delta"]
        self.delta = self.init_delta
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        self.input_size = input_size

        "I suppose pred length is output size? "
        self.output_size = output_size

        "Need also timesteps dt"
        self.dt = params["dt"]

        # self.prev_sol = np.zeros((self.pred_len, self.input_size)) -> TODO find out what predlen is



    def calc_action(self, state):
        return None
