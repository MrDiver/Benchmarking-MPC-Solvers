from MPCBenchmark.agents import Agent, CEM, MPPI, ILQR
from MPCBenchmark.envs import Environment, PendulumEnv, CartPoleSwingUpEnv, AcrobotEnv
from MPCBenchmark.models import Model, PendulumModel, CartPoleSwingUpModel, AcrobotModel
from MPCBenchmark.ExperimentCore import Experiment, Plot, DBTools
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient


def generate_data():
    client = MongoClient("192.168.0.101", 27017)
    db = client.parameter_tuning
    cem_ratio_collection = db.cem_ratios
    mppi_sample_collection = db.mppi_samples


    env_configs = [#(PendulumEnv, PendulumModel, np.array([[np.pi, 0]])),
                   #(CartPoleSwingUpEnv, CartPoleSwingUpModel, np.array([[0, 0, np.pi, 0]])),
                   (AcrobotEnv, AcrobotModel, np.array([[0, 0, 0, 0]]))]


    for env, model, start_states in env_configs:
        for start_state in start_states:
            for K in [10, 20, 50, 100, 200]:
                for T in [5, 10, 25, 50]:
                    #testing the different ratios for cem
                    for ratio in [0.1, 0.25, 0.5, 0.75, 1]:
                        params_cem = {"K": K, "T": T, "max_iter": 10, "n_elite": int(K*ratio), "epsilon": 1e-5, "alpha": 0.2, "std": 1}
                        config = {"env": env, "model": model, "agent": CEM,
                                  "agent_config": params_cem, "experiment_length": 100, "start_state": start_state}
                        for i in range(5):
                            exp = Experiment(config)
                            result = exp(50)
                            result["ratio"] = ratio
                            prepared = DBTools.encodeDict(result)
                            cem_ratio_collection.insert_one(prepared)
                    params_mppi = {"K": K, "T": T, "std": 1, "lam": 0.5}
                    config = {"env": env, "model": model, "agent": MPPI,
                              "agent_config": params_mppi, "experiment_length": 100, "start_state": start_state}
                    for i in range(5):
                        exp = Experiment(config)
                        result = exp(50)
                        prepared = DBTools.encodeDict(result)
                        mppi_sample_collection.insert_one(prepared)

def generate_plots():
    client = MongoClient("localhost", 27017)
    db = client.parameter_tuning
    collection = db.sample_exp

if __name__ == '__main__':
    generate_data()