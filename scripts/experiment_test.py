from MPCBenchmark.agents import Agent, CEM, MPPI, ILQR
from MPCBenchmark.envs import Environment, PendulumEnv, CartPoleSwingUpEnv, AcrobotEnv
from MPCBenchmark.models import Model, PendulumModel, CartPoleSwingUpModel, AcrobotModel
from MPCBenchmark.ExperimentCore import Experiment


agent_config = params_cem = {"K": 50, "T": 15, "max_iter": 5,
                             "n_elite": 5, "epsilon": 1e-5, "alpha": 0.2, "std": 1}
config = {"env": PendulumEnv, "model": PendulumModel, "agent": CEM, }

Experiment()
