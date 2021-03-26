from MPCBenchmark.agents import Agent, CEM, MPPI, ILQR
from MPCBenchmark.envs import Environment, PendulumEnv, CartPoleSwingUpEnv, AcrobotEnv
from MPCBenchmark.models import Model, PendulumModel, CartPoleSwingUpModel, AcrobotModel
from MPCBenchmark.ExperimentCore import Experiment, Plot, DBTools
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
import os
import pprint


def generate_data():
    client = MongoClient("192.168.0.101", 27017)
    db = client.parameter_tuning
    collection = db.noise_tests

    env_configs = [(PendulumEnv, PendulumModel, np.array([[np.pi, 0]]), 0.5, 0.5, 5),
                   (CartPoleSwingUpEnv, CartPoleSwingUpModel, 50,
                    np.array([[0, 0, np.pi, 0]]), 0.1, 0.1, 5),
                   (AcrobotEnv, AcrobotModel, np.array([[0, 0, 0, 0]]), 0.5, 0.1, 25)]

    for env, model, start_states, temp, ratio, max_iter in env_configs:
        for start_state in start_states:
            for T in [5, 10, 25, 50]:
                solver_configs = [(CEM, {"K": 50, "T": T, "max_iter": 10, "n_elite": 50*ratio, "epsilon": 1e-5, "alpha": 0.2, "std": 1}),
                                  (MPPI, {"K": 50, "T": T,
                                          "std": 1, "lam": temp}),
                                  (ILQR, {"T": T, "max_iter": max_iter, "threshold": 1e-7, "closed_loop": False})]
                for solver, solver_config in solver_configs:
                    for actuation_noise in [0.1, 0.5, 1]:
                        config = {"env": env, "model": model, "agent": solver,
                                  "agent_config": solver_config, "experiment_length": 100, "start_state": start_state, "actuation_noise": actuation_noise}
                        for i in range(5):
                            exp = Experiment(config)
                            result = exp(50)
                            prepared = DBTools.encodeDict(result)
                            collection.insert_one(prepared)
                    for sensor_noise in [0.1, 0.5, 1]:
                        config = {"env": env, "model": model, "agent": solver,
                                  "agent_config": solver_config, "experiment_length": 100, "start_state": start_state, "sensor_noise": sensor_noise, "model_noise": True}
                        for i in range(5):
                            exp = Experiment(config)
                            result = exp(50)
                            prepared = DBTools.encodeDict(result)
                            collection.insert_one(prepared)


def generate_plots():
    import matplotlib
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['figure.figsize'] = [20, 12]
    matplotlib.rcParams['legend.fontsize'] = 16
    matplotlib.rcParams['axes.titlesize'] = 22
    matplotlib.rcParams['axes.labelsize'] = 22

    alpha_val = 0.2
    client = MongoClient("localhost", 27017)
    db = client.parameter_tuning
    collection = db.temperature_exp
    for env, statesize in [("PendulumEnvironment", 2), ("CartpoleSwingupEnvironment", 4), ("AcrobotEnvironment", 4)]:
        if not os.path.exists("paper"):
            os.mkdir("paper")
        if not os.path.exists("paper/mppi_temperature"):
            os.mkdir("paper/mppi_temperature")
        if not os.path.exists("paper/mppi_temperature/" + env):
            os.mkdir("paper/mppi_temperature/" + env)


if __name__ == '__main__':
    generate_data()
