# chaos
from MPCBenchmark.envs.gym_wrapper import GymWrapperEnv as GEW
# from MPCBenchmark.envs.mujym_wrapper import MujymWrapperEnv as MEW
from MPCBenchmark.models.gym_model import GymEnvModel as GEM
from MPCBenchmark.agents.cem import CEM
from MPCBenchmark.agents.mppi import MPPI
from MPCBenchmark.agents.ilqr import ILQR
from MPCBenchmark.agents.agent import Agent

from MPCBenchmark.envs.pendulum_env import PendulumEnv as PENV
from MPCBenchmark.envs.cartpole_swingup_env import CartPoleSwingUpEnv as CPSUENV
from MPCBenchmark.envs.env import Environment
from MPCBenchmark.models.pendulum_model import PendulumModel as PEMOD
from MPCBenchmark.models.cartpole_swingup_model import CartPoleSwingUpModel as CPSUMOD
from MPCBenchmark.models.model import DummyModel
from MPCBenchmark.models.model import Model
import numpy as np
import gym_cartpole_swingup
import time
from datetime import datetime
import matplotlib.pyplot as plt
import os
import mpld3

# plt.style.use("seaborn-darkgrid")
plt.rcParams.update({'font.size': 20, 'text.usetex': True})

timestring = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

for path in ["experiments", "experiments/"+timestring, "experiments/"+timestring+"/logs", "experiments/"+timestring+"/plots"]:
    if not os.path.exists(path):
        os.mkdir(path)
experiment_path = "experiments/"+timestring


# ENVIRONMENT = "CartPole-v0"
# ENVIRONMENT = "Pendulum-v0"
# ENVIRONMENT = "InvertedPendulum-v2"


env: Environment = PENV()
model: Model = PEMOD()
env: Environment = CPSUENV()
model: Model = CPSUMOD()
#model: Model = DummyModel(4, 1)
# model = GEM(ENVIRONMENT)j


params_cem = {"K": 50, "T": 15, "max_iter": 5,
              "n_elite": 5, "epsilon": 1e-5, "alpha": 0.2, "instant_cost": (lambda x, u: 0), "std": 1}

params_mppi = {"K": 50, "T": 15, "std": 1,
               "terminal_cost": (lambda x: 0), "instant_cost": (lambda x, u: 0),
               "lam": 0.2}

params_ilqr = {"T": 15, "max_iter": 5, "threshold": 1e-5}
cem: CEM = CEM(model, params_cem)
mppi: MPPI = MPPI(model, params_mppi)
ilqr: ILQR = ILQR(model, params_ilqr)


save_plots = True
# experiment_states = [np.array([np.pi, 0]), np.array(
# [np.pi, 1]), np.array([0, 0]), np.array([np.pi/2, 0])]

experiment_states = [np.array([0, 0, np.pi, 0]), np.array(
    [0, 1, np.pi, 0]), np.array([0, 0, 0, 0]), np.array([0, 1, 0, 0])]


for exp_num, reset_state in enumerate(experiment_states, start=1):
    figcomb = plt.figure(figsize=(30, 25))
    comb_ax = figcomb.subplots(nrows=model.state_size+model.action_size+1)

    duration = 200
    goal_state = np.zeros((duration+1, model.state_size+model.action_size))

    if save_plots:
        for i in range(model.state_size):
            comb_ax[i].set_title("$x_"+str(i)+"$")
            comb_ax[i].set_xlabel("Time s")
            comb_ax[i].set_ylabel("State")
            comb_ax[i].plot(goal_state[:, i], color="tab:red",
                            linestyle=(0, (5, 10)))

        for i in range(model.state_size, model.state_size+model.action_size):
            comb_ax[i].set_title("$u_"+str(i)+"$")
            comb_ax[i].set_xlabel("Time s")
            comb_ax[i].set_ylabel("Action")

    for solver in [cem, mppi, ilqr]:
        solver: Agent = solver
        # env.seed(seed)
        print("\n\n\n", solver.name, " now participates in Experiment No.", exp_num)
        env.reset(reset_state)
        solver.reset()
        starttime = time.time()
        passedtime = 0
        currenttime = time.time()

        solver_fig = plt.figure(figsize=(20, 20))
        solver_fig.suptitle("Test No."+str(exp_num))
        solver_ax = solver_fig.subplots(
            nrows=model.state_size+model.action_size+2)

        computation_time = []
        for i in range(duration):
            action = solver.predict_action(
                env.state, goal_state=np.zeros(model.state_size+model.action_size))
            # newstate = model2.predict(env.state, action)
            # print(newstate)
            _, r, done, _ = env.step(action)
            iterationtime = time.time() - currenttime
            passedtime += iterationtime
            currenttime += iterationtime
            iterationtime = np.around(iterationtime, decimals=3)

            print("==================")
            print("Time Passed:", passedtime)
            print("Iteration Time:", iterationtime)
            # print("State",env.state)
            # print("action",action)
            # print("Cost",-r)
            computation_time.append(iterationtime)
            # print(action, "with reward", r)_get_obs
            if not save_plots:
                env.render()
            # env.history.to_csv(str(i)+"log.txt")
            # if done:
                # env.reset()
        passedtime = np.around(passedtime, decimals=3)
        states = np.array([x for x in env.history["state"].to_numpy()])
        actions = np.array([x for x in env.history["action"].to_numpy()])
        costs = np.array([x for x in env.history["cost"].to_numpy()])

        if save_plots:
            # Plotting for solvers without MPC internals
            solver_ax[0].set_title(solver.name+" actions over time | "+str(
                np.sum(computation_time)) + " s"+" | initial state="+str(reset_state))
            for i in range(model.state_size):
                solver_ax[i].plot(states[:, i], label="$x_" +
                                  str(i)+"$", color="tab:orange", marker="o")
                solver_ax[i].plot(goal_state[:, i], linestyle=(
                    0, (5, 10)), color="tab:red", label="Goal $x_"+str(i)+"$")
                solver_ax[i].set_xlabel("Time s")
                solver_ax[i].set_ylabel("State")

                comb_ax[i].plot(states[:, i], label=solver.name)

            for i in range(model.state_size, model.state_size+model.action_size):
                if len(actions.shape) <= 1:
                    actions = actions[:, None]
                i_ = i-model.state_size
                solver_ax[i].plot(actions[:, i_], label="$u_" +
                                  str(i_)+"$", color="tab:green", marker="o")
                solver_ax[i].set_ylim(
                    model.bounds_low*1.1, model.bounds_high*1.1)
                solver_ax[i].set_xlabel("Time s")
                solver_ax[i].set_ylabel("Action")
                comb_ax[i].plot(actions[:, i_], label=solver.name)
                comb_ax[i].set_ylim(model.bounds_low*1.1,
                                    model.bounds_high*1.1)

            solver_ax[-2].plot(costs, label="costs")
            solver_ax[-2].plot(np.zeros(duration),
                               color="tab:red", linestyle=(0, (5, 10)))
            solver_ax[-2].set_xlabel("Time s")
            solver_ax[-2].set_ylabel("Cost")
            comb_ax[-1].plot(costs, label=solver.name)
            comb_ax[-1].plot(np.zeros(duration),
                             color="tab:red", linestyle=(0, (5, 10)))
            comb_ax[-1].set_xlabel("Time s")
            comb_ax[-1].set_ylabel("Cost")

            solver_ax[-1].plot(computation_time,
                               label="Computation Time (s)", marker="h")
            solver_ax[-1].set_xlabel("Time s")
            solver_ax[-1].set_ylabel("Computation time")
            solver_ax[-1].grid(True)

            # save plot without mpc information
            for ax_ in solver_ax:
                ax_.legend(loc="upper left")
            solver_fig.tight_layout()
            solver_fig.savefig(experiment_path+"/plots/S" +
                               str(exp_num)+"_NoPlanning_"+solver.name+"_trajectory")

            # Adding MPC information to the plots
            mpc_xs = solver.planning_state_history
            mpc_us = solver.planning_action_history
            # print(mpc_xs)
            for start_iteration, planned_states in mpc_xs:
                for i in range(model.state_size):
                    solver_ax[i].plot(range(start_iteration, start_iteration + planned_states.shape[0]), planned_states[:, i],
                                      alpha=0.7, linestyle=(0, (1, 1, 4, 1)), zorder=-1)  # ,label="$x_"+str(i)+"$",color="tab:orange")

            for start_iteration, planned_actions in mpc_us:
                for i in range(model.state_size, model.state_size+model.action_size):
                    if len(actions.shape) <= 1:
                        actions = actions[:, None]
                    i_ = i-model.state_size
                    solver_ax[i].plot(range(start_iteration, start_iteration + planned_actions.shape[0]), planned_actions[:, i_],
                                      alpha=0.7, linestyle=(0, (1, 1, 4, 1)), zorder=-1)  # , label="$u_"+str(i_)+"$",color="tab:green")

            # save plot with MPC information
            solver_fig.tight_layout()
            solver_fig.savefig(experiment_path+"/plots/S" +
                               str(exp_num)+"_Planning_"+solver.name+"_trajectory")

            plt.close(solver_fig)
        env.history.to_csv(experiment_path+"/logs/S" +
                           str(exp_num)+"_"+solver.name+"_log.txt")
        print(solver.name, " finished Experiment No.", exp_num)

    if save_plots:
        for solver_ax in comb_ax:
            solver_ax.legend(loc="upper left")
            solver_ax.grid(True)
        figcomb.suptitle("Test No."+str(exp_num) +
                         " | initial state "+str(reset_state))
        figcomb.tight_layout()
        figcomb.savefig(experiment_path+"/plots/CombS"+str(exp_num)+"_results")
        # mpld3.save_html(figcomb,experiment_path+"/plots/CombS"+str(exp_num)+"_results.html")
        plt.close(figcomb)
