from MPCBenchmark.agents import Agent, CEM, MPPI, ILQR
from MPCBenchmark.envs import Environment, PendulumEnv, CartPoleSwingUpEnv, AcrobotEnv
from MPCBenchmark.models import Model, PendulumModel, CartPoleSwingUpModel, AcrobotModel
from MPCBenchmark.ExperimentCore import Experiment, Plot
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient

client = MongoClient("localhost", 27017)

def quickplot(exp, planning=False):
    fig = Plot.plot_experiment(exp, plot_planning=planning)
    title = "T: "+str(exp.agent_config.get("T"))+" K: "+str(
        exp.agent_config.get("K"))+" max_iter: "+str(exp.agent_config.get("max_iter"))
    fig.suptitle(title+" "+exp.experiment_results["name"])
    fig.tight_layout()
    addon = "_planning"if planning else "_no_planning"
    fig.savefig("experiments/ResultPlots/"+str(T)+"_" +
                str(K)+"_"+str(max_iter)+"_"+exp.Agent.name+"_test"+addon+".png")
    plt.close(fig)


for T in [50]:
    for K in [20]:
        for max_iter in [1]:
            params_cem = {"K": K, "T": T, "max_iter": max_iter,
                          "n_elite": 10, "epsilon": 1e-5, "alpha": 0.2, "std": 1}
            params_mppi = {"K": K, "T": T, "std": 1, "lam": 0.01}
            params_ilqr = {"T": T, "max_iter": max_iter, "threshold": 1e-5, "closed_loop":False}
            experiments = []
            for agent, agent_config in [(CEM, params_cem), (MPPI, params_mppi), (ILQR, params_ilqr)]:
                config = {"env": AcrobotEnv, "model": AcrobotModel, "agent": agent,
                          "agent_config": agent_config, "experiment_length": 200, "start_state": np.array([0,0,0,0])}
                exp = Experiment(config)
                result = exp(50)
                #print(result)
                #experiments.append(exp)
                # quickplot(exp, False)
                # quickplot(exp, True)
            # print("Plot combined")
            # comb_fig = Plot.plot_experiments(experiments)
            # comb_fig.suptitle("T: "+str(T)+" K: "+str(K)+" in "+exp.model.name)
            # comb_fig.tight_layout()
            # comb_fig.savefig("experiments/ResultPlots/comb_"+str(T)+"_" +
            #                  str(K)+"_"+str(max_iter)+"_test")
            # plt.close(comb_fig)
