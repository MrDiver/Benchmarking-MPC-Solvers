from MPCBenchmark.agents import Agent, CEM, MPPI, ILQR
from MPCBenchmark.envs import Environment, PendulumEnv, CartPoleSwingUpEnv, AcrobotEnv
from MPCBenchmark.models import Model, PendulumModel, CartPoleSwingUpModel, AcrobotModel
from MPCBenchmark.ExperimentCore import Experiment, Plot, DBTools
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
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
    client = MongoClient("localhost", 27017)
    db = client.parameter_tuning
    collection = db.temperature_exp
    for env, statesize in [("PendulumEnvironment", 2), ("CartpoleSwingupEnvironment", 4), ("AcrobotEnvironment", 2)]:
        ls = [0.001, 0.01, 0.1, 0.5, 1, 10]
        Ts = [5, 10, 25, 50]
        for T in Ts:
            fig_exp = plt.figure(figsize=(8, 8))
            ax_exp = fig_exp.subplots(nrows=statesize)
            for l in ls:
                print(env)
                query = {"agent_config.T": T,
                         "agent_config.lam": l, "env_name": env}
                print(collection.count(query))
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
                # print(state_mean)
                fig = plt.figure(figsize=(20, 12))
                ax = fig.subplots(nrows=statesize+actionsize+1)

                low = state_mean-2*state_std
                high = state_mean+2*state_std
                for i in range(statesize):
                    ax[i].fill_between(range(horizon_length),
                                       low[:, i], high[:, i], alpha=0.5)
                    ax[i].plot(range(horizon_length), state_mean[:, i])
                    for traj in states:
                        ax[i].plot(range(horizon_length),
                                   traj[:, i], alpha=0.6)

                    ax_exp[i].fill_between(range(horizon_length),
                                           low[:, i], high[:, i], alpha=0.2)
                    ax_exp[i].plot(range(horizon_length),
                                   state_mean[:, i], label="l="+str(l))

                low = action_mean-2*action_std
                high = action_mean+2*action_std
                for i, j in enumerate(range(statesize, statesize+actionsize)):
                    ax[j].fill_between(range(horizon_length),
                                       low[:, i], high[:, i], alpha=0.5)
                    ax[j].plot(range(horizon_length), action_mean[:, i])

                low = cost_mean - 2*cost_std.flatten()
                high = cost_mean + 2*cost_std.flatten()
                print(cost_std)
                ax[-1].fill_between(range(horizon_length),
                                    low, high, alpha=0.5)
                ax[-1].plot(cost_mean)
                fig.suptitle("MPPI Temperature "+env +
                             " T:"+str(T)+" lambda:"+str(l))
                fig.tight_layout()
                plt.close(fig)
            fig_exp.suptitle("MPPI Temperature "+env +
                             " T:"+str(T))
            [tmp.legend() for tmp in ax_exp]
            fig_exp.tight_layout()
            plt.show()
    print("Done")


if __name__ == '__main__':
    generate_plots()
