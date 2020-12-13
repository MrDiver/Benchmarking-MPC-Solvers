from MPCBenchmark.envs.gym_wrapper import GymWrapperEnv as GEW
from MPCBenchmark.models.gym_model import GymEnvModel as GEM
from MPCBenchmark.agents.mppi import MPPI
from MPCBenchmark.agents.cem import CEM
import numpy as np
import gym_cartpole_swingup


ENVIRONMENT = "CartPole-v0"
ENVIRONMENT = "Pendulum-v0"
#ENVIRONMENT = "InvertedPendulum-v2"
env = GEW(ENVIRONMENT)
model = GEM(ENVIRONMENT)
params_cem = {"max_iter": 10, "n_samples": 60,
              "n_elite": 10, "epsilon": 1e-6, "alpha": 0.1, "instant_cost": (lambda x, u: 0), "variance": 1}

params_mppi = {"K": 50, "T": 50, "Sigma": np.eye(1)*0.1,
               "terminal_cost": (lambda x: 0), "instant_cost": (lambda x, u: 0),
               "lam": 0.2}

cem = CEM(env.bounds_low, env.bounds_high, 4, 1, model, params_cem)
mppi = MPPI(env.bounds_low, env.bounds_high, 4, 1, model, params_mppi)


for i in range(1000):
    action = mppi.calc_action(env.state)
    _, r, done, _ = env.step(action)
    #print(action, "with reward", r)
    env.render()
    if done:
        env.reset()

print(env.history)
