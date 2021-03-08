from MPCBenchmark.ExperimentCore import Experiment
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 20, 'text.usetex': False})


def plot_experiment(exp: Experiment, figsize=(20, 20), plot_planning=False, fig=None):

    states = exp.experiment_results["env_states"]
    actions = exp.experiment_results["env_actions"]
    costs = exp.experiment_results["env_costs"]
    computation_time = exp.experiment_results["computation_time"]

    solver_fig: fig if fig is not None else plt.Figure = plt.figure(
        figsize=figsize)
    solver_ax = solver_fig.subplots(
        nrows=exp.Model.state_size+exp.Model.action_size+2)
    print(exp.agent.name)
    solver_ax[0].set_title(exp.agent.name+" actions over time | "+str(
        np.sum(computation_time)) + " s"+" | initial state="+str(exp.start_state))
    for i in range(exp.Model.state_size):
        solver_ax[i].plot(states[:, i], label="$x_" +
                          str(i)+"$", color="tab:orange", marker="o")
        # solver_ax[i].plot(goal_state[:, i], linestyle=(
        #    0, (5, 10)), color="tab:red", label="Goal $x_"+str(i)+"$")
        solver_ax[i].set_xlabel("Time s")
        solver_ax[i].set_ylabel("State")

    for i in range(exp.Model.state_size, exp.Model.state_size+exp.Model.action_size):
        if len(actions.shape) <= 1:
            actions = actions[:, None]
        i_ = i-exp.Model.state_size
        solver_ax[i].plot(actions[:, i_], label="$u_" +
                          str(i_)+"$", color="tab:green", marker="o")
        solver_ax[i].set_ylim(
            exp.Model.bounds_low*1.1, exp.Model.bounds_high*1.1)
        solver_ax[i].set_xlabel("Time s")
        solver_ax[i].set_ylabel("Action")

    solver_ax[-2].plot(costs, label="costs")
    solver_ax[-2].plot(np.zeros(exp.experiment_length),
                       color="tab:red", linestyle=(0, (5, 10)))
    solver_ax[-2].set_xlabel("Time s")
    solver_ax[-2].set_ylabel("Cost")

    solver_ax[-1].plot(computation_time,
                       label="Computation Time (s)", marker="h")
    solver_ax[-1].set_xlabel("Time s")
    solver_ax[-1].set_ylabel("Computation time")
    solver_ax[-1].grid(True)

    # save plot without mpc information
    for ax_ in solver_ax:
        ax_.legend(loc="upper left")
    solver_fig.tight_layout()
    # solver_fig.savefig(experiment_path+"/plots/S" +
    #                   str(exp_num)+"_NoPlanning_"+solver.name+"_trajectory")

    if plot_planning:
        # Adding MPC information to the plots
        mpc_xs = exp.experiment_results["agent_planning_states"]
        mpc_us = exp.experiment_results["agent_planning_actions"]
        # print(mpc_xs)
        for start_iteration, planned_states in mpc_xs:
            for i in range(exp.Model.state_size):
                solver_ax[i].plot(range(start_iteration, start_iteration + planned_states.shape[0]), planned_states[:, i],
                                  alpha=0.7, linestyle=(0, (1, 1, 4, 1)), zorder=-1)  # ,label="$x_"+str(i)+"$",color="tab:orange")

        for start_iteration, planned_actions in mpc_us:
            for i in range(exp.Model.state_size, exp.Model.state_size+exp.Model.action_size):
                if len(actions.shape) <= 1:
                    actions = actions[:, None]
                i_ = i-exp.Model.state_size
                solver_ax[i].plot(range(start_iteration, start_iteration + planned_actions.shape[0]), planned_actions[:, i_],
                                  alpha=0.7, linestyle=(0, (1, 1, 4, 1)), zorder=-1)  # , label="$u_"+str(i_)+"$",color="tab:green")

        # save plot with MPC information
        solver_fig.tight_layout()
        # solver_fig.savefig(experiment_path+"/plots/S" +
        #               str(exp_num)+"_Planning_"+solver.name+"_trajectory")

    return solver_fig
