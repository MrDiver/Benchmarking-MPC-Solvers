# chaos
from MPCBenchmark.envs.gym_wrapper import GymWrapperEnv as GEW
# from MPCBenchmark.envs.mujym_wrapper import MujymWrapperEnv as MEW
from MPCBenchmark.models.gym_model import GymEnvModel as GEM
from MPCBenchmark.agents.mppi import MPPI
from MPCBenchmark.agents.cem import CEM
from MPCBenchmark.agents.ilqr import ILQR
from MPCBenchmark.envs.gym_pendulum_env import PendulumEnv as PENV
import numpy as np
import gym_cartpole_swingup


#ENVIRONMENT = "CartPole-v0"
ENVIRONMENT = "Pendulum-v0"
#ENVIRONMENT = "InvertedPendulum-v2"
#ENVIRONMENT = "CartPoleSwingUp-v0"
#env = GEW(ENVIRONMENT)


env = PENV()
model = GEM(ENVIRONMENT)
params_cem = {"K": 50, "T": 50, "max_iter": 1,
              "n_elite": 5, "epsilon": 1e-5, "alpha": 0, "instant_cost": (lambda x, u: 0), "std": 1}

params_mppi = {"K": 50, "T": 50, "std": 1,
               "terminal_cost": (lambda x: 0), "instant_cost": (lambda x, u: 0),
               "lam": 0.2}
params_ilqr = {"max_iter": 1, "init_mu": 50, "mu_min": 0, "mu_max": 60, "init_delta": 0.1, "threshold": np.pi,
               "terminal_cost": (lambda x: 0), "input_cost": (lambda x, u: 0),
               "state_cost": (lambda x: 0)}

cem = CEM(env.bounds_low, env.bounds_high, 4, 1, model, params_cem)
mppi = MPPI(env.bounds_low, env.bounds_high, 4, 1, model, params_mppi)
ilqr = ILQR(env.bounds_low, env.bounds_high, 4, 1, model, params_ilqr)


env.reset()
for i in range(1000):
    action = cem.calc_action(env.state)
    _, r, done, _ = env.step(action)
    # print(action, "with reward", r)_get_obs
    env.render()
    #if done:
        #env.reset()

print(env.history)
