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
    client = MongoClient("localhost", 27017)
    db = client.parameter_tuning
    collection = db.temperature_exp

    env_configs = [(PendulumEnv, PendulumModel, np.array([[np.pi, 0]])),
                   (CartPoleSwingUpEnv, CartPoleSwingUpModel,
                    np.array([[0, 0, np.pi, 0]])),
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
    plt.rcParams.update({'font.size': 10, 'text.usetex': False})
    client = MongoClient("192.168.0.101", 27017)
    db = client.parameter_tuning
    collection = db.temperature_exp
    for env, statesize in [("PendulumEnvironment", 2), ("CartpoleSwingupEnvironment", 4), ("AcrobotEnvironment", 4)]:
        if not os.path.exists("paper"):
            os.mkdir("paper")
        if not os.path.exists("paper/mppi_temperature"):
            os.mkdir("paper/mppi_temperature")
        if not os.path.exists("paper/mppi_temperature/" + env):
            os.mkdir("paper/mppi_temperature/" + env)

        ls = [0.001, 0.01, 0.1, 0.5, 1, 10]
        Ts = [5, 10, 25, 50]

        T_perf = []
        print(env)

        for T in Ts:
            fig_exp = plt.figure(figsize=(20, 12))
            ax_exp = fig_exp.subplots(nrows=statesize+2)
            lam_perf = []
            for l in ls:
                query = {"agent_config.T": T,
                         "agent_config.lam": l, "env_name": env}
                states = []
                actions = []
                costs = []
                for result in collection.find(query):
                    tmp = DBTools.decodeDict(result)
                    states.append(tmp["env_states"])
                    actions.append(tmp["env_actions"])
                    costs.append(tmp["env_costs"])
                    # pprint.pprint(tmp["name"])
                state_mean = np.mean(states, axis=0)
                state_std = np.std(states, axis=0)
                action_mean = np.mean(actions, axis=0)
                action_std = np.std(actions, axis=0)
                cost_mean = np.mean(costs, axis=0)
                cost_std = np.std(actions, axis=0)
                horizon_length = state_mean.shape[0]
                statesize = state_mean.shape[1]
                actionsize = action_mean.shape[1]

                #Add to list
                lam_perf.append(cost_mean.sum())
                # print(state_mean)
                fig = plt.figure(figsize=(20, 12))
                ax = fig.subplots(nrows=statesize+actionsize+1)

                low = state_mean-2*state_std
                high = state_mean+2*state_std
                #plot states
                for i in range(statesize):
                    ax[i].fill_between(range(horizon_length),
                                       low[:, i], high[:, i], alpha=0.5, label="l="+str(l))
                    ax[i].plot(range(horizon_length), state_mean[:, i], label="l="+str(l))
                    ax[i].set_ylabel("x_"+str(i))
                    for traj in states:
                        ax[i].plot(range(horizon_length),
                                   traj[:, i], alpha=0.2)

                    ax_exp[i].fill_between(range(horizon_length),
                                           low[:, i], high[:, i], alpha=0.2)
                    ax_exp[i].plot(range(horizon_length),
                                   state_mean[:, i], label="l="+str(l))
                    ax_exp[i].set_ylabel("x_" + str(i))

                low = action_mean-2*action_std
                high = action_mean+2*action_std
                for i, j in enumerate(range(statesize, statesize+actionsize)):
                    ax[j].fill_between(range(horizon_length),
                                       low[:, i], high[:, i], alpha=0.2)
                    ax[j].plot(range(horizon_length), action_mean[:, i], label="l="+str(l))
                    ax[j].set_ylabel("u_"+str(i))
                    ax_exp[j].fill_between(range(horizon_length),
                                       low[:, i], high[:, i], alpha=0.2)
                    ax_exp[j].plot(range(horizon_length), action_mean[:, i], label="l=" + str(l))
                    ax_exp[j].set_ylabel("u_" + str(i))

                low = cost_mean - 2*cost_std.flatten()
                high = cost_mean + 2*cost_std.flatten()
                ax[-1].fill_between(range(horizon_length),
                                    low, high, alpha=0.5)
                ax[-1].plot(cost_mean, label="l=" + str(l))

                ax_exp[-1].fill_between(range(horizon_length),
                                    low, high, alpha=0.5)
                ax_exp[-1].plot(cost_mean, label="l=" + str(l))

                for ax in ax:
                    ax.legend(loc="upper left")
                fig.suptitle("MPPI Temperature "+env +
                             " T:"+str(T)+" lambda:"+str(l))
                fig.tight_layout()

                plt.savefig("paper/mppi_temperature/"+env+"/"+str(T)+"_mppi_l"+str(l).replace(".", "_")+".png")
                plt.close(fig)
            T_perf.append(lam_perf)

            fig_exp.suptitle("MPPI Temperature "+env +
                             " T:"+str(T))
            [tmp.legend(loc="upper left") for tmp in ax_exp]
            fig_exp.tight_layout()
            plt.savefig("paper/mppi_temperature/"+env+"/"+str(T)+"_mppi_combined.png")
            plt.close(fig_exp)

        T_perf = np.array(T_perf)
        fig_perf = plt.figure(figsize=(20, 12))
        ax_perf = fig_perf.add_subplot()
        for i in range(len(ls)):
            ax_perf.plot(Ts, T_perf[:, i], label="l "+str(ls[i]))
        fig_perf.suptitle("MPPI Temperature Summary "+env)
        fig_perf.savefig("paper/mppi_temperature/"+env+"/mppi_summary.png")
        ax_perf.set_xticks(Ts)
        ax_perf.set_xlabel("Horizon Length - T")
        ax_perf.set_ylabel("Performance Cost")
        ax_perf.legend(loc="upper left")
        plt.show()
        plt.close(fig_perf)
    print("Done")


if __name__ == '__main__':
    generate_plots()
