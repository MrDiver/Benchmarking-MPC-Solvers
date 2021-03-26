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
    collection = db.ilqr_runs2

    env_configs = [#(PendulumEnv, PendulumModel, np.array([[np.pi, 0]])),
                   #(CartPoleSwingUpEnv, CartPoleSwingUpModel,
                   # np.array([[0, 0, np.pi, 0]])),
                   (AcrobotEnv, AcrobotModel, np.array([[0, 0, 0, 0]]))]

    for env, model, start_states in env_configs:
        for start_state in start_states:
            for max_iter in [5, 10, 25]:
                for T in [5, 10, 25, 50]:
                    params_ilqr = {"T": T, "max_iter": max_iter, "threshold": 1e-7, "closed_loop": False}
                    config = {"env": env, "model": model, "agent": ILQR,
                              "agent_config": params_ilqr, "experiment_length": 100, "start_state": start_state}
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
    client = MongoClient("192.168.0.101", 27017)
    db = client.parameter_tuning
    collection = db.ilqr_runs2
    for env, statesize in [("PendulumEnvironment", 2), ("CartpoleSwingupEnvironment", 4), ("AcrobotEnvironment", 4)]:
        if not os.path.exists("paper"):
            os.mkdir("paper")
        if not os.path.exists("paper/ilqr_iterations"):
            os.mkdir("paper/ilqr_iterations")
        if not os.path.exists("paper/ilqr_iterations/" + env):
            os.mkdir("paper/ilqr_iterations/" + env)

        M_perf_median = []
        M_perf_25th = []
        M_perf_75th = []
        Ts = [5, 10, 25, 50]
        max_iters = [5, 10, 25]
        for max_iter in max_iters:
            fig_exp = plt.figure()
            ax_exp = fig_exp.subplots(nrows=statesize + 2)
            T_perf_median = []
            T_perf_25th = []
            T_perf_75th = []

            for T in Ts:
                query = {"agent_config.T": T,
                         "agent_config.max_iter": max_iter,
                         "agent_config.closed_loop": False,
                         "env_name": env}
                print(query)
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

                # Add to list
                tmp_sum = np.sum(costs, axis=1)
                T_perf_median.append(np.median(tmp_sum, axis=0))
                T_perf_25th.append(np.quantile(tmp_sum, 0.25, axis=0))
                T_perf_75th.append(np.quantile(tmp_sum, 0.75, axis=0))
                # print(state_mean)
                fig = plt.figure()
                ax = fig.subplots(nrows=statesize + actionsize + 1)

                low = state_mean - 2 * state_std
                high = state_mean + 2 * state_std
                # plot states
                label_l = "T=" + str(T)
                for i in range(statesize):
                    ax[i].fill_between(range(horizon_length),
                                       low[:, i], high[:, i], alpha=alpha_val, label=label_l)
                    ax[i].plot(range(horizon_length), state_mean[:, i], label=label_l)
                    ax[i].set_ylabel("x_" + str(i))
                    for traj in states:
                        ax[i].plot(range(horizon_length),
                                   traj[:, i], alpha=alpha_val)

                    ax_exp[i].fill_between(range(horizon_length),
                                           low[:, i], high[:, i], alpha=alpha_val)
                    ax_exp[i].plot(range(horizon_length),
                                   state_mean[:, i], label=label_l)
                    ax_exp[i].set_ylabel("x_" + str(i))

                low = action_mean - 2 * action_std
                high = action_mean + 2 * action_std
                for i, j in enumerate(range(statesize, statesize + actionsize)):
                    ax[j].fill_between(range(horizon_length),
                                       low[:, i], high[:, i], alpha=alpha_val)
                    ax[j].plot(range(horizon_length), action_mean[:, i], label=label_l)
                    ax[j].set_ylabel("u_" + str(i))
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
                fig.suptitle("ILQR Iteration Performance " + env +
                             " T:" + str(T) + " Maximum Iterations:" + str(max_iter))
                fig.tight_layout()

                plt.savefig(
                    "paper/ilqr_iterations/" + env + "/" + str(T) + "_ilqr_m" + str(max_iter) + ".png")
                plt.close(fig)

            fig_exp.suptitle("ILQR Iterations " + env +
                             " Maximum Iteration:" + str(max_iter))
            [tmp.legend(loc="upper left") for tmp in ax_exp]
            fig_exp.tight_layout()
            fig_exp.savefig("paper/ilqr_iterations/" + env + "/combined" + str(max_iter) + "_ilqr.png")
            # extent = ax_exp[-1].get_window_extent().transformed(fig_exp.dpi_scale_trans.inverted())
            # fig_exp.savefig("paper/mppi_temperature/"+env+"/"+str(T)+"_mppi_combined_cost_only.png", bbox_inches=extent.expanded(1.1, 1.2))
            plt.close(fig_exp)

            M_perf_median.append(T_perf_median)
            M_perf_25th.append(T_perf_25th)
            M_perf_75th.append(T_perf_75th)

        M_perf_median = np.array(M_perf_median)
        M_perf_25th = np.array(M_perf_25th)
        M_perf_75th = np.array(M_perf_75th)

        fig_perf = plt.figure()
        ax_perf = fig_perf.add_subplot()
        for i in range(len(max_iters)):
            ax_perf.plot(Ts, M_perf_median[i], label="Max Iterations:" + str(max_iters[i]))
            low = M_perf_25th[i]
            high = M_perf_75th[i]
            ax_perf.fill_between(Ts, low, high, alpha=alpha_val)
        fig_perf.suptitle("ILQR Iterations Summary " + env)
        ax_perf.set_xticks(Ts)
        ax_perf.set_xlabel("Horizon Length - T")
        ax_perf.set_ylabel("Performance Cost")
        ax_perf.legend(loc="upper left")
        ax_perf.grid()
        plt.show()
        fig_perf.savefig("paper/ilqr_iterations/" + env + "_ilqr_summary.png")
        plt.close(fig_perf)


if __name__ == '__main__':
    generate_plots()
