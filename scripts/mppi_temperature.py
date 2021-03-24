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

        ls = [0.001, 0.01, 0.1, 0.5, 1, 10]
        Ts = [5, 10, 25, 50]

        T_perf_median = []
        T_perf_25th = []
        T_perf_75th = []
        T_perf_ratios = []
        print(env)

        for T in Ts:
            fig_exp = plt.figure()
            ax_exp = fig_exp.subplots(nrows=statesize+2)
            lam_perf_median = []
            lam_perf_25th = []
            lam_perf_75th = []
            lam_ratios = []
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
                costs = np.clip(costs, 0, 20)
                state_mean = np.mean(states, axis=0)
                state_std = np.std(states, axis=0)
                action_mean = np.mean(actions, axis=0)
                action_std = np.std(actions, axis=0)
                cost_median = np.median(costs, axis=0)
                cost_25th = np.quantile(costs, 0.25, axis=0).flatten()
                cost_75th = np.quantile(costs, 0.75, axis=0).flatten()
                horizon_length = state_mean.shape[0]
                statesize = state_mean.shape[1]
                actionsize = action_mean.shape[1]

                #Add to list
                tmp_sum = np.sum(costs, axis=1)
                lam_perf_median.append(np.median(tmp_sum, axis=0))
                lam_perf_25th.append(np.quantile(tmp_sum, 0.25, axis=0))
                lam_perf_75th.append(np.quantile(tmp_sum, 0.75, axis=0))
                # print(state_mean)
                fig = plt.figure()
                ax = fig.subplots(nrows=statesize+actionsize+1)

                low = state_mean-2*state_std
                high = state_mean+2*state_std
                #plot states
                ratio = np.round(l/np.mean(costs), 4)
                lam_ratios.append(ratio)
                label_l = "l="+str(l) + " r="+str(ratio)
                for i in range(statesize):
                    ax[i].fill_between(range(horizon_length),
                                       low[:, i], high[:, i], alpha=alpha_val, label=label_l)
                    ax[i].plot(range(horizon_length), state_mean[:, i], label=label_l)
                    ax[i].set_ylabel("x_"+str(i))
                    for traj in states:
                        ax[i].plot(range(horizon_length),
                                   traj[:, i], alpha=alpha_val)

                    ax_exp[i].fill_between(range(horizon_length),
                                           low[:, i], high[:, i], alpha=alpha_val)
                    ax_exp[i].plot(range(horizon_length),
                                   state_mean[:, i], label=label_l)
                    ax_exp[i].set_ylabel("x_" + str(i))

                low = action_mean-2*action_std
                high = action_mean+2*action_std
                for i, j in enumerate(range(statesize, statesize+actionsize)):
                    ax[j].fill_between(range(horizon_length),
                                       low[:, i], high[:, i], alpha=alpha_val)
                    ax[j].plot(range(horizon_length), action_mean[:, i], label=label_l)
                    ax[j].set_ylabel("u_"+str(i))
                    ax_exp[j].fill_between(range(horizon_length),
                                       low[:, i], high[:, i], alpha=alpha_val)
                    ax_exp[j].plot(range(horizon_length), action_mean[:, i], label=label_l)
                    ax_exp[j].set_ylabel("u_" + str(i))

                low = cost_25th
                high = cost_75th
                ax[-1].fill_between(range(horizon_length),
                                    low, high, alpha=alpha_val)
                ax[-1].plot(cost_median, label=label_l)
                ax[-1].set_ylabel("Costs")
                ax[-1].set_xlabel("Time - t")
                ax_exp[-1].fill_between(range(horizon_length),
                                    low, high, alpha=alpha_val)
                ax_exp[-1].plot(cost_median, label=label_l)
                ax_exp[-1].set_ylabel("Costs")
                ax_exp[-1].set_xlabel("Time - t")

                for ax in ax:
                    ax.legend(loc="upper left")
                fig.suptitle("MPPI Temperature "+env +
                             " T:"+str(T)+" lambda:"+str(l))
                fig.tight_layout()

                plt.savefig("paper/mppi_temperature/"+env+"/"+str(T)+"_mppi_l"+str(l).replace(".", "_")+".png")
                plt.close(fig)
            T_perf_median.append(lam_perf_median)
            T_perf_25th.append(lam_perf_25th)
            T_perf_75th.append(lam_perf_75th)
            T_perf_ratios.append(lam_ratios)

            fig_exp.suptitle("MPPI Temperature "+env +
                             " T:"+str(T))
            [tmp.legend(loc="upper left") for tmp in ax_exp]
            fig_exp.tight_layout()
            fig_exp.savefig("paper/mppi_temperature/"+env+"/combined"+str(T)+"_mppi.png")
            # extent = ax_exp[-1].get_window_extent().transformed(fig_exp.dpi_scale_trans.inverted())
            # fig_exp.savefig("paper/mppi_temperature/"+env+"/"+str(T)+"_mppi_combined_cost_only.png", bbox_inches=extent.expanded(1.1, 1.2))
            plt.close(fig_exp)

        T_perf_median = np.array(T_perf_median)
        T_perf_25th = np.array(T_perf_25th)
        T_perf_75th = np.array(T_perf_75th)
        T_perf_ratios = np.array(T_perf_ratios)
        ratios = np.round(np.mean(T_perf_ratios,axis=0),4)
        print("25", T_perf_25th.T)
        print("50", T_perf_median.T)
        print("75", T_perf_75th.T)
        fig_perf = plt.figure()
        ax_perf = fig_perf.add_subplot()
        for i in range(len(ls)):
            ax_perf.plot(Ts, T_perf_median[:, i], label="l="+str(ls[i]) + " r="+str(ratios[i]))
            low = T_perf_25th[:, i]
            high =T_perf_75th[:, i]
            ax_perf.fill_between(Ts,low,high,alpha=alpha_val)
        fig_perf.suptitle("MPPI Temperature Summary "+env)
        ax_perf.set_xticks(Ts)
        ax_perf.set_xlabel("Horizon Length - T")
        ax_perf.set_ylabel("Performance Cost")
        ax_perf.legend(loc="upper left")
        ax_perf.grid()
        plt.show()
        fig_perf.savefig("paper/mppi_temperature/"+env+"_mppi_summary.png")
        plt.close(fig_perf)
    print("Done")


if __name__ == '__main__':
    generate_plots()
