from MPCBenchmark.envs.gym_wrapper import GymWrapperEnv as GEW
import gym_cartpole_swingup


ENVIRONMENT = "CartPoleSwingUp-v0"
env = GEW(ENVIRONMENT)

for i in range(1000):
    env.step(0)
    env.render()
