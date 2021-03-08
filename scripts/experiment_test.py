from MPCBenchmark.agents import Agent, CEM, MPPI, ILQR
from MPCBenchmark.envs import Environment, PendulumEnv, CartPoleSwingUpEnv, AcrobotEnv
from MPCBenchmark.models import Model, PendulumModel, CartPoleSwingUpModel, AcrobotModel
from MPCBenchmark.ExperimentCore import Experiment, Plot
import numpy as np
import matplotlib.pyplot as plt

params_cem = {"K": 50, "T": 15, "max_iter": 5,
              "n_elite": 20, "epsilon": 1e-5, "alpha": 0.2, "std": 1}
params_mppi = {"K": 50, "T": 15, "std": 1, "lam": 0.8}
params_ilqr = {"T": 15, "max_iter": 5, "threshold": 1e-5}


for agent, agent_config in [(CEM, params_cem), (MPPI, params_mppi), (ILQR, params_ilqr)]:
    config = {"env": PendulumEnv, "model": PendulumModel, "agent": agent,
              "agent_config": agent_config, "experiment_length": 50, "start_state": np.array([np.pi, 0])}
    exp = Experiment(config)
    result = exp()
    fig = Plot.plot_experiment(exp)
plt.show()
