from MPCBenchmark.agents import Agent, CEM, MPPI, ILQR
from MPCBenchmark.envs import Environment, PendulumEnv, CartPoleSwingUpEnv, AcrobotEnv
from MPCBenchmark.models import Model, PendulumModel, CartPoleSwingUpModel, AcrobotModel
from MPCBenchmark.ExperimentCore import Experiment, Plot
import numpy as np
import matplotlib.pyplot as plt


def quickplot(exp, planning=False):
    fig = Plot.plot_experiment(exp, plot_planning=planning)
    title = "T: "+str(exp.agent_config.get("T"))+" K: "+str(
        exp.agent_config.get("K"))+" max_iter: "+str(exp.agent_config.get("max_iter"))
    fig.suptitle(title+" "+exp.experiment_results["name"])
    fig.tight_layout()
    addon = "_planning"if planning else "_no_planning"
    fig.savefig("experiments/paper/"+str(T)+"_" +
                str(K)+"_"+str(max_iter)+"_"+exp.Agent.name+"_test"+addon)
    plt.close(fig)


for T in [15, 30, 60, 100]:
    for K in [50, 100, 200]:
        for max_iter in [5, 10, 20]:
            params_cem = {"K": K, "T": T, "max_iter": max_iter,
                          "n_elite": 20, "epsilon": 1e-5, "alpha": 0.2, "std": 1}
            params_mppi = {"K": K, "T": T, "std": 1, "lam": 0.1}
            params_ilqr = {"T": T, "max_iter": max_iter, "threshold": 1e-5}
            experiments = []
            for agent, agent_config in [(CEM, params_cem), (MPPI, params_mppi), (ILQR, params_ilqr)]:
                config = {"env": PendulumEnv, "model": PendulumModel, "agent": agent,
                          "agent_config": agent_config, "experiment_length": 100, "start_state": np.array([np.pi, 0])}
                exp = Experiment(config)
                result = exp()
                experiments.append(exp)
                quickplot(exp, False)
                quickplot(exp, True)
            print("Plot combined")
            comb_fig = Plot.plot_experiments(experiments)
            comb_fig.suptitle("T: "+str(T)+" K: "+str(K)+" in "+exp.model.name)
            comb_fig.tight_layout()
            comb_fig.savefig("experiments/paper/comb_"+str(T)+"_" +
                             str(K)+"_"+str(max_iter)+"_test")
            plt.close(comb_fig)
plt.show()
