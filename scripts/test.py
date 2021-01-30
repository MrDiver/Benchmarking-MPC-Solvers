from MPCBenchmark.envs.gym_wrapper import GymWrapperEnv as GEW
from MPCBenchmark.envs.mujym_wrapper import MujymWrapperEnv as MEW
from MPCBenchmark.models.gym_model import GymEnvModel as GEM
from MPCBenchmark.agents.mppi import MPPI
from MPCBenchmark.agents.cem import CEM
import numpy as np
import gym_cartpole_swingup


ENVIRONMENT = "CartPole-v0"
ENVIRONMENT = "Pendulum-v0"
ENVIRONMENT = "InvertedPendulum-v2"
ENVIRONMENT = "CartPoleSwingUp-v0"
env = GEW(ENVIRONMENT)
model = GEM(ENVIRONMENT)
params_cem = {"K": 25, "T": 25, "max_iter": 5,
              "n_elite": 20, "epsilon": 1e-5, "alpha": 0.2, "instant_cost": (lambda x, u: 0), "std": 1}

params_mppi = {"K": 25, "T": 25, "std": 1,
               "terminal_cost": (lambda x: 3*x[1]**2), "instant_cost": (lambda x, u: 0),
               "lam": 0.2}

cem = CEM(env.bounds_low, env.bounds_high, 4, 1, model, params_cem)
mppi = MPPI(env.bounds_low, env.bounds_high, 4, 1, model, params_mppi)


for i in range(1000):
    action = cem.calc_action(env.state)
    _, r, done, _ = env.step(action)
    # print(action, "with reward", r)
    env.render()
    if done:
        env.reset()

print(env.history)
