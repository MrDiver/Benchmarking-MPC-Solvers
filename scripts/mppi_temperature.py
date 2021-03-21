from MPCBenchmark.agents import Agent, CEM, MPPI, ILQR
from MPCBenchmark.envs import Environment, PendulumEnv, CartPoleSwingUpEnv, AcrobotEnv
from MPCBenchmark.models import Model, PendulumModel, CartPoleSwingUpModel, AcrobotModel
from MPCBenchmark.ExperimentCore import Experiment, Plot, DBTools
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient


def generate_data():
    client = MongoClient("localhost", 27017)
    db = client.parameter_tuning
    collection = db.temperature_exp


    env_configs = [(PendulumEnv, PendulumModel, np.array([[np.pi, 0]])),
                   (CartPoleSwingUpEnv, CartPoleSwingUpModel, np.array([[0, 0, np.pi, 0]])),
                   (AcrobotEnv, AcrobotModel, np.array([[0, 0, 0, 0]]))]


    for env, model, start_states in env_configs:
        for start_state in start_states:
            for l in [0.001, 0.01, 0.1, 0.5, 1, 10]:
                for T in [5, 10, 25, 50]:
                    params_mppi = {"K": 100, "T": T, "std": 1, "lam": l}
                    config = {"env": env, "model": model, "agent": MPPI,
                              "agent_config": params_mppi, "experiment_length": 100, "start_state": start_state}
                    for i in range(5):
                        exp = Experiment(config)
                        result = exp(50)
                        prepared = DBTools.encodeDict(result)
                        collection.insert_one(prepared)

def generate_plots():
    client = MongoClient("localhost", 27017)
    db = client.parameter_tuning
    collection = db.temperature_exp

if __name__ == '__main__':
    generate_data()