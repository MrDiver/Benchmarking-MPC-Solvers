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
    collection = db.noise_tests

    env_configs = [(PendulumEnv, PendulumModel, np.array([[np.pi, 0]]), 0.5, 0.5, 5),
                   (CartPoleSwingUpEnv, CartPoleSwingUpModel,
                   np.array([[0, 0, np.pi, 0]]), 0.1, 0.1, 5),
                   (AcrobotEnv, AcrobotModel, np.array([[0, 0, 0, 0]]), 0.5, 0.1, 25)
                   ]


    for env, model, start_states, temp, ratio, max_iter in env_configs:
        for start_state in start_states:
            for T in [5, 10, 25, 50]:
                solver_configs = [(CEM, {"K": 50, "T": T, "max_iter": 10, "n_elite": int(50*ratio), "epsilon": 1e-5, "alpha": 0.2, "std": 1}),
                                  (MPPI, {"K": 50, "T": T,
                                          "std": 1, "lam": temp}),
                                  (ILQR, {"T": T, "max_iter": max_iter, "threshold": 1e-7, "closed_loop": False})]
                for solver, solver_config in solver_configs:
                    for actuation_noise in [0.1, 0.5, 1]:
                        print("Actuation", actuation_noise)
                        config = {"env": env, "model": model, "agent": solver,
                                  "agent_config": solver_config, "experiment_length": 100, "start_state": start_state, "actuation_noise": actuation_noise}
                        for i in range(5):
                            exp = Experiment(config)
                            result = exp(50)
                            prepared = DBTools.encodeDict(result)
                            collection.insert_one(prepared)
                    for sensor_noise in [0.1, 0.5, 1]:
                        print("Sensor", sensor_noise)
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
    client = MongoClient("192.168.0.101", 27017)
    db = client.parameter_tuning
    collection = db.noise_tests
    true_collections = [db.cem_ratios, db.ilqr_runs2, db.mppi_samples, db.temperature_exp]

    """
    ###############################
    
        Actuation Noise Plots
    
    ###############################
    """
    env_configs = [("PendulumEnvironment", 2, 0.5, 0.5, 5, 50),
                   ("CartpoleSwingupEnvironment", 4, 0.1, 0.1, 5, 10),
                   ("AcrobotEnvironment", 4, 0.5, 0.1, 25, 10)]

    for env, statesize,temp, ratio, max_iter, T in env_configs:
        if not os.path.exists("paper"):
            os.mkdir("paper")
        if not os.path.exists("paper/noise_test"):
            os.mkdir("paper/noise_test")
        if not os.path.exists("paper/noise_test/" + env):
            os.mkdir("paper/noise_test/" + env)
        if not os.path.exists("paper/noise_test/" + env + "/actuation_noise"):
            os.mkdir("paper/noise_test/" + env + "/actuation_noise")

        # Generate Actuation Noise Plots
        solver_median = []
        solver_25th = []
        solver_75th = []

        noise_values = [0, 0.1, 0.5, 1.0]
        fig_comb = plt.figure()
        ax_comb = fig_comb.add_subplot()

        for solver in ["CEM","MPPI","ILQR"]:
            noise_median = []
            noise_25th = []
            noise_75th = []

            query = {"agent_name": solver,
                     "env_name": env,
                     "agent_config.T":T,
                     }
            if solver == "CEM":
                query["ratio"] = ratio
                query["agent_config.K"] = 50
            if solver == "MPPI":
                query["agent_config.lam"] = temp
                query["agent_config.K"] = 100
            if solver == "ILQR":
                query["agent_config.max_iter"] = max_iter

            tmp_costs = []
            #getting true system
            for tmp_colletion in true_collections:
                for i, result in enumerate(tmp_colletion.find(query)):
                    tmp = DBTools.decodeDict(result)
                    tmp_costs.append(np.sum(tmp["env_costs"]))
            tmp_median = np.median(tmp_costs)
            tmp_25th = np.quantile(tmp_costs, 0.25)
            tmp_75th = np.quantile(tmp_costs, 0.75)
            noise_median.append(tmp_median)
            noise_25th.append(tmp_25th)
            noise_75th.append(tmp_75th)

            for actuation_noise in [0.1, 0.5, 1]:
                query = {"agent_name": solver,
                         "actuation_noise_std": actuation_noise,
                         "agent_config.T": T,
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

                noise_median.append(np.sum(cost_median))
                noise_25th.append(np.sum(cost_25th))
                noise_75th.append(np.sum(cost_75th))

                fig = plt.figure()
                ax = fig.subplots(nrows=statesize + actionsize + 1)

                low = state_mean - 2 * state_std
                high = state_mean + 2 * state_std
                label_l = "noise=" + str(actuation_noise)
                for i in range(statesize):
                    ax[i].fill_between(range(horizon_length),
                                       low[:, i], high[:, i], alpha=alpha_val, label=label_l)
                    ax[i].plot(range(horizon_length), state_mean[:, i], label=label_l)
                    ax[i].set_ylabel("x_" + str(i))
                    for traj in states:
                        ax[i].plot(range(horizon_length),
                                   traj[:, i], alpha=alpha_val)

                low = action_mean - 2 * action_std
                high = action_mean + 2 * action_std
                for i, j in enumerate(range(statesize, statesize + actionsize)):
                    # Noised values
                    ax[j].fill_between(range(horizon_length),
                                       low[:, i], high[:, i], alpha=alpha_val)
                    ax[j].plot(range(horizon_length), action_mean[:, i], label=label_l)
                    ax[j].set_ylabel("u_" + str(i))

                low = cost_25th
                high = cost_75th
                ax[-1].fill_between(range(horizon_length),
                                    low, high, alpha=alpha_val)
                ax[-1].plot(cost_median, label=label_l)
                ax[-1].set_ylabel("Costs")
                ax[-1].set_xlabel("Time - t")

                for ax in ax:
                    ax.legend(loc="upper left")
                fig.suptitle(str(solver)+" Actuation Noise Performance " + env + " stddev:" + str(actuation_noise))
                fig.tight_layout()

                plt.savefig(
                    "paper/noise_test/" + env + "/actuation_noise/" + str(solver) +"_averaged_n" +str(actuation_noise).replace(".", "_")+ ".png")
                plt.close(fig)

            solver_median.append(noise_median)
            solver_25th.append(noise_25th)
            solver_75th.append(noise_75th)
            # Plot
            fig_noise = plt.figure()
            ax_noise = fig_noise.add_subplot()

            ax_noise.plot(noise_values, noise_median, label=solver)
            ax_comb.plot(noise_values, noise_median, label=solver)
            low = noise_25th
            high = noise_75th
            ax_noise.fill_between(noise_values, low, high, alpha=alpha_val)
            ax_comb.fill_between(noise_values, low, high, alpha=alpha_val, edgecolor="black")
            ax_noise.set_xticks(noise_values)
            ax_noise.set_xlabel("Noise Amount - std")
            ax_noise.grid()
            ax_noise.set_ylabel("Performance Cost")
            ax_noise.legend(loc="upper left")
            fig_noise.suptitle(solver + " noise test " + env)

            fig_noise.savefig("paper/noise_test/" + env + "/actuation_noise/" + str(solver)+"_summary.png")
            plt.close(fig_noise)

        ax_comb.set_xticks(noise_values)
        ax_noise.set_yticks(np.arange(0,10,20).tolist())
        ax_comb.set_xlabel("Noise Amount - std")
        ax_comb.grid()
        ax_comb.set_ylabel("Performance Cost")
        ax_comb.legend(loc="upper left")
        fig_comb.suptitle("Actuation Noise Test Summary " + env + " | T =" +str(T))
        fig_comb.savefig("paper/noise_test/" + env + "_actuation_noise_combined_summary.png")

    """
    ###############################

        Sensor Noise Plots

    ###############################
    """

    for env, statesize, temp, ratio, max_iter, T in env_configs:
        if not os.path.exists("paper"):
            os.mkdir("paper")
        if not os.path.exists("paper/noise_test"):
            os.mkdir("paper/noise_test")
        if not os.path.exists("paper/noise_test/" + env):
            os.mkdir("paper/noise_test/" + env)
        if not os.path.exists("paper/noise_test/" + env + "/sensor_noise"):
            os.mkdir("paper/noise_test/" + env + "/sensor_noise")

        # Generate Actuation Noise Plots
        solver_median = []
        solver_25th = []
        solver_75th = []

        noise_values = [0, 0.1, 0.5, 1.0]
        fig_comb = plt.figure()
        ax_comb = fig_comb.add_subplot()

        for solver in ["CEM", "MPPI", "ILQR"]:
            noise_median = []
            noise_25th = []
            noise_75th = []

            query = {"agent_name": solver,
                     "env_name": env,
                     "agent_config.T": T,
                     }
            if solver == "CEM":
                query["ratio"] = ratio
                query["agent_config.K"] = 50
            if solver == "MPPI":
                query["agent_config.lam"] = temp
                query["agent_config.K"] = 100
            if solver == "ILQR":
                query["agent_config.max_iter"] = max_iter

            tmp_costs = []
            # getting true system
            for tmp_colletion in true_collections:
                for i, result in enumerate(tmp_colletion.find(query)):
                    tmp = DBTools.decodeDict(result)
                    tmp_costs.append(np.sum(tmp["env_costs"]))
            tmp_median = np.median(tmp_costs)
            tmp_25th = np.quantile(tmp_costs, 0.25)
            tmp_75th = np.quantile(tmp_costs, 0.75)
            noise_median.append(tmp_median)
            noise_25th.append(tmp_25th)
            noise_75th.append(tmp_75th)

            for actuation_noise in [0.1, 0.5, 1]:
                query = {"agent_name": solver,
                         "sensor_noise_std": actuation_noise,
                         "agent_config.T": T,
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

                noise_median.append(np.sum(cost_median))
                noise_25th.append(np.sum(cost_25th))
                noise_75th.append(np.sum(cost_75th))

                fig = plt.figure()
                ax = fig.subplots(nrows=statesize + actionsize + 1)

                low = state_mean - 2 * state_std
                high = state_mean + 2 * state_std
                label_l = "noise=" + str(actuation_noise)
                for i in range(statesize):
                    ax[i].fill_between(range(horizon_length),
                                       low[:, i], high[:, i], alpha=alpha_val, label=label_l)
                    ax[i].plot(range(horizon_length), state_mean[:, i], label=label_l)
                    ax[i].set_ylabel("x_" + str(i))
                    for traj in states:
                        ax[i].plot(range(horizon_length),
                                   traj[:, i], alpha=alpha_val)

                low = action_mean - 2 * action_std
                high = action_mean + 2 * action_std
                for i, j in enumerate(range(statesize, statesize + actionsize)):
                    # Noised values
                    ax[j].fill_between(range(horizon_length),
                                       low[:, i], high[:, i], alpha=alpha_val)
                    ax[j].plot(range(horizon_length), action_mean[:, i], label=label_l)
                    ax[j].set_ylabel("u_" + str(i))

                low = cost_25th
                high = cost_75th
                ax[-1].fill_between(range(horizon_length),
                                    low, high, alpha=alpha_val)
                ax[-1].plot(cost_median, label=label_l)
                ax[-1].set_ylabel("Costs")
                ax[-1].set_xlabel("Time - t")

                for ax in ax:
                    ax.legend(loc="upper left")
                fig.suptitle(str(solver) + " Sensor Noise Performance " + env + " stddev:" + str(actuation_noise))
                fig.tight_layout()

                plt.savefig(
                    "paper/noise_test/" + env + "/sensor_noise/" + str(solver) + "_averaged_n" + str(
                        actuation_noise).replace(".", "_") + ".png")
                plt.close(fig)

            solver_median.append(noise_median)
            solver_25th.append(noise_25th)
            solver_75th.append(noise_75th)
            # Plot
            fig_noise = plt.figure()
            ax_noise = fig_noise.add_subplot()

            ax_noise.plot(noise_values, noise_median, label=solver)
            ax_comb.plot(noise_values, noise_median, label=solver)
            low = noise_25th
            high = noise_75th
            ax_noise.fill_between(noise_values, low, high, alpha=alpha_val)
            ax_comb.fill_between(noise_values, low, high, alpha=alpha_val, edgecolor="black")
            ax_noise.set_xticks(noise_values)
            ax_noise.set_xlabel("Noise Amount - std")
            ax_noise.grid()
            ax_noise.set_ylabel("Performance Cost")
            ax_noise.legend(loc="upper left")
            fig_noise.suptitle(solver + " noise test " + env)

            fig_noise.savefig("paper/noise_test/" + env + "/sensor_noise/" + str(solver) + "_summary.png")
            plt.close(fig_noise)

        ax_comb.set_xticks(noise_values)
        ax_noise.set_yticks(np.arange(0, 10, 20).tolist())
        ax_comb.set_xlabel("Noise Amount - std")
        ax_comb.grid()
        ax_comb.set_ylabel("Performance Cost")
        ax_comb.legend(loc="upper left")
        fig_comb.suptitle("Sensor Noise Test Summary " + env + " | T =" +str(T))
        fig_comb.savefig("paper/noise_test/" + env + "_sensor_noise_combined_summary.png")


if __name__ == '__main__':
    generate_plots()
