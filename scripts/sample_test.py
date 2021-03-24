from MPCBenchmark.agents import Agent, CEM, MPPI, ILQR
from MPCBenchmark.envs import Environment, PendulumEnv, CartPoleSwingUpEnv, AcrobotEnv
from MPCBenchmark.models import Model, PendulumModel, CartPoleSwingUpModel, AcrobotModel
from MPCBenchmark.ExperimentCore import Experiment, Plot, DBTools
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
import os

def generate_data():
    client = MongoClient("localhost", 27017)
    db = client.parameter_tuning
    cem_ratio_collection = db.cem_ratios
    mppi_sample_collection = db.mppi_samples


    env_configs = [(PendulumEnv, PendulumModel, np.array([[np.pi, 0]])),
                   (CartPoleSwingUpEnv, CartPoleSwingUpModel, np.array([[0, 0, np.pi, 0]])),
                   #(AcrobotEnv, AcrobotModel, np.array([[0, 0, 0, 0]]))
                   ]



    generate_cem = False
    generate_mppi = True
    for env, model, start_states in env_configs:
        for start_state in start_states:
            for K in [10,20,50,100,200]:
                for T in [5, 10, 25, 50]:
                    #testing the different ratios for cem
                    if generate_cem:
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
                    #generate data for mppi
                    if generate_mppi:
                        params_mppi = {"K": K, "T": T, "std": 1, "lam": 0.5}
                        config = {"env": env, "model": model, "agent": MPPI,
                                  "agent_config": params_mppi, "experiment_length": 100, "start_state": start_state}
                        for i in range(5):
                            exp = Experiment(config)
                            result = exp(50)
                            prepared = DBTools.encodeDict(result)
                            mppi_sample_collection.insert_one(prepared)





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
    cem_ratio_collection = db.cem_ratios
    mppi_sample_collection = db.mppi_samples

    Ts = [5, 10, 25, 50]
    Ks = [10, 20, 50, 100, 200]
    ratios = [0.1, 0.25, 0.5, 0.75, 1]

    generate_single_exps = True
    generate_combined_exps = True
    generate_summaries = True
    generate_cem = False
    generate_mppi = True

    if generate_cem:
        for env, statesize in [("PendulumEnvironment", 2), ("CartpoleSwingupEnvironment", 4), ("AcrobotEnvironment", 4)]:
            if not os.path.exists("paper"):
                os.mkdir("paper")
            if not os.path.exists("paper/cem_ratio"):
                os.mkdir("paper/cem_ratio")
            if not os.path.exists("paper/cem_ratio/" + env):
                os.mkdir("paper/cem_ratio/" + env)


            """
            =============================================
            
                    Cem Ratio Test Plot Generation
                
            =============================================
            """
            T_perf_median = []
            T_perf_25th = []
            T_perf_75th = []
            for T in Ts:
                K_perf_median = []
                K_perf_25th = []
                K_perf_75th = []
                for K in Ks:
                    ratio_perf_median = []
                    ratio_perf_25th = []
                    ratio_perf_75th = []
                    fig_exp = plt.figure(figsize)
                    ax_exp = fig_exp.subplots(nrows=statesize + 2)
                    for ratio in ratios:
                        query = {"agent_config.T": T,
                                 "agent_config.K": K, "ratio":ratio, "env_name": env}

                        states = []
                        actions = []
                        costs = []
                        for result in cem_ratio_collection.find(query):
                            tmp = DBTools.decodeDict(result)
                            states.append(tmp["env_states"])
                            actions.append(tmp["env_actions"])
                            costs.append(tmp["env_costs"])
                            # print(tmp["name"])
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

                        # add to list
                        tmp_sum = np.sum(costs, axis=1)
                        ratio_perf_median.append(np.median(tmp_sum, axis=0))
                        ratio_perf_25th.append(np.quantile(tmp_sum, 0.25, axis=0))
                        ratio_perf_75th.append(np.quantile(tmp_sum, 0.75, axis=0))

                        # Plot the experiments for the given T,K,ratio
                        fig = plt.figure(figsize)
                        ax = fig.subplots(nrows=statesize + actionsize + 1)

                        low = state_mean - 2 * state_std
                        high = state_mean + 2 * state_std
                        # plot states
                        label_l = "r=" + str(ratio)
                        for i in range(statesize):
                            if generate_single_exps:
                                ax[i].fill_between(range(horizon_length),
                                                   low[:, i], high[:, i], alpha=alpha_val, label=label_l)
                                ax[i].plot(range(horizon_length), state_mean[:, i], label=label_l)
                                ax[i].set_ylabel("x_" + str(i))
                                for traj in states:
                                    ax[i].plot(range(horizon_length),
                                               traj[:, i], alpha=alpha_val)

                            if generate_combined_exps:
                                ax_exp[i].fill_between(range(horizon_length),
                                                       low[:, i], high[:, i], alpha=alpha_val)
                                ax_exp[i].plot(range(horizon_length),
                                               state_mean[:, i], label=label_l)
                                ax_exp[i].set_ylabel("x_" + str(i))

                        low = action_mean - 2 * action_std
                        high = action_mean + 2 * action_std
                        for i, j in enumerate(range(statesize, statesize + actionsize)):
                            if generate_single_exps:
                                ax[j].fill_between(range(horizon_length),
                                                   low[:, i], high[:, i], alpha=alpha_val)
                                ax[j].plot(range(horizon_length), action_mean[:, i], label=label_l)
                                ax[j].set_ylabel("u_" + str(i))
                            if generate_combined_exps:
                                ax_exp[j].fill_between(range(horizon_length),
                                                       low[:, i], high[:, i], alpha=alpha_val)
                                ax_exp[j].plot(range(horizon_length), action_mean[:, i], label=label_l)
                                ax_exp[j].set_ylabel("u_" + str(i))

                        low = cost_25th
                        high = cost_75th
                        if generate_single_exps:
                            ax[-1].fill_between(range(horizon_length),
                                                low, high, alpha=alpha_val)
                            ax[-1].plot(cost_median, label=label_l)
                            ax[-1].set_ylabel("Costs")
                            ax[-1].set_xlabel("Time - t")
                            for ax in ax:
                                ax.legend(loc="upper left")
                            fig.suptitle("CEM Ratio " + env +
                                         " T:" + str(T) + " K:" + str(K) + " r:" +str(ratio))
                            fig.tight_layout()

                            plt.savefig(
                                "paper/cem_ratio/" + env + "/" + str(T) + "_" + str(K) +"_cem_r" + str(ratio).replace(".", "_") + ".png")

                        if generate_combined_exps:
                            ax_exp[-1].fill_between(range(horizon_length),
                                                    low, high, alpha=alpha_val)
                            ax_exp[-1].plot(cost_median, label=label_l)
                            ax_exp[-1].set_ylabel("Costs")
                            ax_exp[-1].set_xlabel("Time - t")


                        plt.close(fig)

                    K_perf_median.append(ratio_perf_median)
                    K_perf_25th.append(ratio_perf_25th)
                    K_perf_75th.append(ratio_perf_75th)

                    if generate_combined_exps:
                        fig_exp.suptitle("Cem Ratios " + env +
                                         " T:" + str(T) + " K:" + str(K))
                        [tmp.legend(loc="upper left") for tmp in ax_exp]
                        fig_exp.tight_layout()
                        print(T,K,env)
                        fig_exp.savefig("paper/cem_ratio/" + env + "/combined" + str(T) + "_" + str(K) + "_cem.png")
                        # extent = ax_exp[-1].get_window_extent().transformed(fig_exp.dpi_scale_trans.inverted())
                        # fig_exp.savefig("paper/mppi_temperature/"+env+"/"+str(T)+"_mppi_combined_cost_only.png", bbox_inches=extent.expanded(1.1, 1.2))
                    plt.close(fig_exp)

                K_perf_median = np.array(K_perf_median)
                K_perf_25th = np.array(K_perf_25th)
                K_perf_75th = np.array(K_perf_75th)

                T_perf_median.append(K_perf_median)
                T_perf_25th.append(K_perf_25th)
                T_perf_75th.append(K_perf_75th)

                if generate_summaries:
                    fig_perf = plt.figure(figsize)
                    ax_perf = fig_perf.add_subplot()
                    for i in range(len(ratios)):
                        ax_perf.plot(Ks, K_perf_median[:, i], label=" r=" + str(ratios[i]))
                        low = K_perf_25th[:, i]
                        high = K_perf_75th[:, i]
                        ax_perf.fill_between(Ks, low, high, alpha=alpha_val)
                    fig_perf.suptitle("CEM Ratio Summary " + env + " for T:"+str(T))
                    ax_perf.set_xticks(Ks)
                    ax_perf.set_xlabel("Sample Amount - K")
                    ax_perf.set_ylabel("Performance Cost")
                    ax_perf.legend(loc="upper left")
                    ax_perf.grid()
                    plt.show()
                    fig_perf.savefig("paper/cem_ratio/" + env + "_"+str(T)+"_cem_summary.png")
                    plt.close(fig_perf)

            T_perf_median = np.mean(T_perf_median, axis=0)
            T_perf_25th = np.mean(T_perf_25th, axis=0)
            T_perf_75th = np.mean(T_perf_75th, axis=0)

            fig_perf = plt.figure(figsize)
            ax_perf = fig_perf.add_subplot()
            for i in range(len(ratios)-1):
                ax_perf.plot(Ks, T_perf_median[:, i], label=" r=" + str(ratios[i]))
                low = T_perf_25th[:, i]
                high = T_perf_75th[:, i]
                ax_perf.fill_between(Ks, low, high, alpha=alpha_val)
            fig_perf.suptitle("CEM Ratios Summary " + env + " averaged over Horizon Length")
            ax_perf.set_xticks(Ks)
            ax_perf.set_xlabel("Sample Amount - K")
            ax_perf.set_ylabel("Performance Cost")
            ax_perf.legend(loc="upper left")
            ax_perf.grid()
            plt.show()
            fig_perf.savefig("paper/cem_ratio/Final_" + env + "_cem_summary.png")
            plt.close(fig_perf)

        """
        =============================================

                     MPPI sample test

        =============================================
        """

    Ks = [10, 20, 50, 100, 200, 500]
    if generate_mppi:
        for env, statesize in [("PendulumEnvironment", 2), ("CartpoleSwingupEnvironment", 4), ("AcrobotEnvironment", 4)]:
            if not os.path.exists("paper"):
                os.mkdir("paper")
            if not os.path.exists("paper/mppi_samples"):
                os.mkdir("paper/mppi_samples")
            if not os.path.exists("paper/mppi_samples/" + env):
                os.mkdir("paper/mppi_samples/" + env)

            T_perf_median = [] #(Ts,K_perf)
            T_perf_25th = []
            T_perf_75th = []
            for T in Ts:
                K_perf_median = [] #(K_perf)
                K_perf_25th = []
                K_perf_75th = []
                fig_exp = plt.figure()
                ax_exp = fig_exp.subplots(nrows=statesize + 2)
                for K in Ks:
                    query = {"agent_config.T": T,
                             "agent_config.K": K, "env_name": env}
                    print(query)
                    states = []
                    actions = []
                    costs = []
                    for result in mppi_sample_collection.find(query):
                        tmp = DBTools.decodeDict(result)
                        states.append(tmp["env_states"])
                        actions.append(tmp["env_actions"])
                        costs.append(tmp["env_costs"])
                        # print(tmp["name"])
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

                    # add to list
                    tmp_sum = np.sum(costs, axis=1)
                    K_perf_median.append(np.median(tmp_sum, axis=0))
                    K_perf_25th.append(np.quantile(tmp_sum, 0.25, axis=0))
                    K_perf_75th.append(np.quantile(tmp_sum, 0.75, axis=0))

                    # Plot the experiments for the given T,K,ratio
                    fig = plt.figure()
                    ax = fig.subplots(nrows=statesize + actionsize + 1)

                    low = state_mean - 2 * state_std
                    high = state_mean + 2 * state_std
                    # plot states
                    label_l = "K:"+str(K)
                    for i in range(statesize):
                        if generate_single_exps:
                            ax[i].fill_between(range(horizon_length),
                                               low[:, i], high[:, i], alpha=alpha_val, label=label_l)
                            ax[i].plot(range(horizon_length), state_mean[:, i], label=label_l)
                            ax[i].set_ylabel("x_" + str(i))
                            for traj in states:
                                ax[i].plot(range(horizon_length),
                                           traj[:, i], alpha=alpha_val)

                        if generate_combined_exps:
                            ax_exp[i].fill_between(range(horizon_length),
                                                   low[:, i], high[:, i], alpha=alpha_val)
                            ax_exp[i].plot(range(horizon_length),
                                           state_mean[:, i], label=label_l)
                            ax_exp[i].set_ylabel("x_" + str(i))

                    low = action_mean - 2 * action_std
                    high = action_mean + 2 * action_std
                    for i, j in enumerate(range(statesize, statesize + actionsize)):
                        if generate_single_exps:
                            ax[j].fill_between(range(horizon_length),
                                               low[:, i], high[:, i], alpha=alpha_val)
                            ax[j].plot(range(horizon_length), action_mean[:, i], label=label_l)
                            ax[j].set_ylabel("u_" + str(i))
                        if generate_combined_exps:
                            ax_exp[j].fill_between(range(horizon_length),
                                                   low[:, i], high[:, i], alpha=alpha_val)
                            ax_exp[j].plot(range(horizon_length), action_mean[:, i], label=label_l)
                            ax_exp[j].set_ylabel("u_" + str(i))

                    low = cost_25th
                    high = cost_75th
                    if generate_single_exps:
                        ax[-1].fill_between(range(horizon_length),
                                            low, high, alpha=alpha_val)
                        ax[-1].plot(cost_median, label=label_l)
                        ax[-1].set_ylabel("Costs")
                        ax[-1].set_xlabel("Time - t")
                        for ax in ax:
                            ax.legend(loc="upper left")
                        fig.suptitle("MPPI Ratio " + env +
                                     " T:" + str(T) + " K:" + str(K))
                        fig.tight_layout()

                        plt.savefig(
                            "paper/mppi_samples/" + env + "/" + str(T) + "_" + str(K) + "_mppi" + ".png")

                    if generate_combined_exps:
                        ax_exp[-1].fill_between(range(horizon_length),
                                                low, high, alpha=alpha_val)
                        ax_exp[-1].plot(cost_median, label=label_l)

                    plt.close(fig)

                if generate_combined_exps:
                    fig_exp.suptitle("MPPI Samples " + env +
                                     " T:" + str(T))
                    ax_exp[-1].set_ylabel("Costs")
                    ax_exp[-1].set_xlabel("Time - t")
                    [tmp.legend(loc="upper left") for tmp in ax_exp]
                    fig_exp.tight_layout()
                    print(T, env)
                    fig_exp.savefig("paper/mppi_samples/" + env + "/combined" + str(T) + "_mppi.png")
                    # extent = ax_exp[-1].get_window_extent().transformed(fig_exp.dpi_scale_trans.inverted())
                    # fig_exp.savefig("paper/mppi_temperature/"+env+"/"+str(T)+"_mppi_combined_cost_only.png", bbox_inches=extent.expanded(1.1, 1.2))
                plt.close(fig_exp)

                K_perf_median = np.array(K_perf_median)
                K_perf_25th = np.array(K_perf_25th)
                K_perf_75th = np.array(K_perf_75th)

                T_perf_median.append(K_perf_median)
                T_perf_25th.append(K_perf_25th)
                T_perf_75th.append(K_perf_75th)

            if generate_summaries:
                fig_perf = plt.figure()
                ax_perf = fig_perf.add_subplot()
                #T_perf_median = np.mean(T_perf_median,axis=0)
                #T_perf_25th = np.mean(T_perf_25th, axis=0)
                #T_perf_75th = np.mean(T_perf_75th, axis=0)
                for i in range(len(Ts)):
                    ax_perf.plot(Ks, T_perf_median[i], label=" T=" + str(Ts[i]))
                    low = T_perf_25th[i]
                    high = T_perf_75th[i]
                    ax_perf.fill_between(Ks, low, high, alpha=alpha_val)
                fig_perf.suptitle("MPPI Sample Summary " + env)
                ax_perf.set_xticks(Ks)
                ax_perf.set_xlabel("Sample Amount - K")
                ax_perf.set_ylabel("Performance Cost")
                ax_perf.legend(loc="upper left")
                ax_perf.grid()
                plt.show()
                fig_perf.savefig("paper/mppi_samples/" + env + "_mppi_summary.png")
                plt.close(fig_perf)



if __name__ == '__main__':
    generate_plots()