from MPCBenchmark.agents import Agent, CEM, MPPI, ILQR
from MPCBenchmark.envs import Environment, PendulumEnv, CartPoleSwingUpEnv, AcrobotEnv
from MPCBenchmark.models import Model, PendulumModel, CartPoleSwingUpModel, AcrobotModel
from MPCBenchmark.ExperimentCore import Experiment, Plot, DBTools
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
import os
import pprint


def generate_plots():
    import matplotlib
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['figure.figsize'] = [20, 12]
    matplotlib.rcParams['legend.fontsize'] = 16
    matplotlib.rcParams['axes.titlesize'] = 22
    matplotlib.rcParams['axes.labelsize'] = 22

    alpha_val = 0.2
    client = MongoClient("192.168.0.101", 27017)
    db = client.parameter_tuning
    collections = [db.cem_ratios, db.ilqr_runs2, db.mppi_samples, db.temperature_exp]

    for env, statesize in [("PendulumEnvironment", 2), ("CartpoleSwingupEnvironment", 4), ("AcrobotEnvironment", 4)]:
        if not os.path.exists("paper"):
            os.mkdir("paper")
        if not os.path.exists("paper/time_comparison"):
            os.mkdir("paper/time_comparison")
        if not os.path.exists("paper/time_comparison/" + env):
            os.mkdir("paper/time_comparison/" + env)

        fig = plt.figure()
        ax = fig.subplots(2,2)
        ax = ax.flatten()


        for solver in ["CEM","MPPI","ILQR"]:
            for i,T in enumerate([5,10,25,50]):
                times = []
                costs = []
                query = {"env_name":  env, "agent_name":solver, "agent_config.T":T}
                for collection in collections:
                    for result in collection.find(query):
                        tmp = DBTools.decodeDict(result)
                        times.append(tmp["passed_time"])
                        costs.append(tmp["env_costs"])
                costs = np.clip(costs, -20, 20)
                costs = np.sum(costs, axis=1)
                ax[i].scatter(costs, times, label=solver)
                ax[i].legend()
                ax[i].set_title("T:"+str(T))
                ax[i].set_ylabel("Time")
                ax[i].set_xlabel("Performance")

        fig.suptitle(env)
        fig.tight_layout()

        plt.show()



if __name__ == '__main__':
    generate_plots()
