# chaos
from MPCBenchmark.envs.gym_wrapper import GymWrapperEnv as GEW
# from MPCBenchmark.envs.mujym_wrapper import MujymWrapperEnv as MEW
from MPCBenchmark.models.gym_model import GymEnvModel as GEM

from MPCBenchmark.agents.cem import CEM
from MPCBenchmark.agents.mppi import MPPI
from MPCBenchmark.agents.ilqr import ILQR
from MPCBenchmark.agents.ilqr2 import ILQR as ILQR2

from MPCBenchmark.envs.gym_pendulum_env import PendulumEnv as PENV
from MPCBenchmark.models.gym_pendulum_model import PendulumModel as PEMOD
import numpy as np
import gym_cartpole_swingup
import time
from datetime import datetime
import matplotlib.pyplot as plt
import os
import mpld3

#plt.style.use("seaborn-darkgrid")
plt.rcParams.update({'font.size': 20,'text.usetex': True})

timestring = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

for path in ["experiments","experiments/"+timestring,"experiments/"+timestring+"/logs","experiments/"+timestring+"/plots"]:
    if not os.path.exists(path):
        os.mkdir(path)
experiment_path = "experiments/"+timestring


#ENVIRONMENT = "CartPole-v0"
ENVIRONMENT = "Pendulum-v0"
#ENVIRONMENT = "InvertedPendulum-v2"


env = PENV()
model = PEMOD()
#model = GEM(ENVIRONMENT)j


params_cem = {"K": 50, "T": 15, "max_iter": 5,
              "n_elite": 5, "epsilon": 1e-5, "alpha": 0.2, "instant_cost": (lambda x, u: 0), "std": 1}

params_mppi = {"K": 50, "T": 15, "std": 1,
               "terminal_cost": (lambda x: 0), "instant_cost": (lambda x, u: 0),
               "lam": 0.2}
params_ilqr = {"T": 50, "max_iter": 1, "init_mu": 50, "mu_min": 0, "mu_max": 60, "init_delta": 0.1, "threshold": np.pi,
               "terminal_cost": (lambda x: 0), "input_cost": (lambda x, u: 0),
               "state_cost": (lambda x: 0)}

params_ilqr2 = {"T":15, "max_iter":5, "threshold":1e-5}
state_size = 2
output_size = 1
cem = CEM(env.bounds_low, env.bounds_high, state_size, output_size, model, params_cem)
mppi = MPPI(env.bounds_low, env.bounds_high, state_size, output_size, model, params_mppi)
ilqr = ILQR(env.bounds_low, env.bounds_high, state_size, output_size, model, params_ilqr)
ilqr2 = ILQR2(env.bounds_low,env.bounds_high, state_size, output_size, model, params_ilqr2)



save_plots = True

for exp_num,reset_state in enumerate([np.array([np.pi,0]),np.array([np.pi,1]),np.array([0,0]),np.array([np.pi/2,0])],start=1):
    figcomb = plt.figure(figsize=(30,20))
    comb_ax = figcomb.subplots(nrows=state_size+output_size)

    duration = 150
    goal_state = np.zeros((duration+1,2))

    for i in range(state_size):
        comb_ax[i].set_title("$x_"+str(i)+"$")
        comb_ax[i].set_xlabel("Time s")
        comb_ax[i].set_ylabel("State")
        comb_ax[i].plot(goal_state[:,i],color="tab:red",linestyle=(0, (5, 10)))

    for i in range(state_size,state_size+output_size):
        comb_ax[i].set_title("$u_"+str(i)+"$")
        comb_ax[i].set_xlabel("Time s")
        comb_ax[i].set_ylabel("Action")

    for solver in [cem,mppi,ilqr2]:
        #env.seed(seed)
        print("\n\n\n",solver.name," now participates in Experiment No.",exp_num)
        env.reset(reset_state)
        solver.reset()
        starttime = time.time()
        passedtime = 0
        currenttime = time.time()

        solver_fig = plt.figure(figsize=(20,20))
        solver_fig.suptitle("Test No."+str(exp_num))
        solver_ax = solver_fig.subplots(nrows=state_size+output_size+2)

        computation_time = []
        for i in range(duration):
            action = solver.calc_action(env.state, goal_state=[0,0,0])
            #newstate = model2.predict(env.state, action)
            #print(newstate)
            _, r, done, _ = env.step(action)
            iterationtime = time.time() - currenttime
            passedtime += iterationtime
            currenttime+=iterationtime
            iterationtime = np.around(iterationtime,decimals=3)

            print("==================")
            print("Time Passed:",passedtime)
            print("Iteration Time:",iterationtime)
            # print("State",env.state)
            # print("action",action)
            # print("Cost",-r)
            computation_time.append(iterationtime)
            # print(action, "with reward", r)_get_obs
            if not save_plots:
                env.render()
            #env.history.to_csv(str(i)+"log.txt")
            if done:
                env.reset()
        passedtime = np.around(passedtime,decimals=3)
        states = np.array([x for x in env.history["state"].to_numpy()])
        actions = np.array([x for x in env.history["action"].to_numpy()])
        costs = np.array([x for x in env.history["cost"].to_numpy()])

        if save_plots:
            #Plotting for solvers without MPC internals
            solver_ax[0].set_title(solver.name+" actions over time | "+str(np.sum(computation_time))+ " s"+" | initial state="+str(reset_state))   
            for i in range(state_size):
                solver_ax[i].plot(states[:,i],label="$x_"+str(i)+"$",color="tab:orange")
                solver_ax[i].plot(goal_state[:,i],linestyle=(0, (5, 10)),color="tab:red",label="Goal $x_"+str(i)+"$")                
                solver_ax[i].set_xlabel("Time s")
                solver_ax[i].set_ylabel("State")

                comb_ax[i].plot(states[:,i],label=solver.name)

            for i in range(state_size,state_size+output_size):
                if len(actions.shape) <=1:
                    actions = actions[:,None]
                i_ = i-state_size
                solver_ax[i].plot(actions[:,i_], label="$u_"+str(i_)+"$",color="tab:green")
                solver_ax[i].set_ylim(model.bounds_low*1.1,model.bounds_high*1.1)
                solver_ax[i].set_xlabel("Time s")
                solver_ax[i].set_ylabel("Action")
                comb_ax[i].plot(actions[:,i_],label=solver.name)
                comb_ax[i].set_ylim(model.bounds_low*1.1,model.bounds_high*1.1)

            solver_ax[-2].plot(costs, label="costs")
            solver_ax[-2].plot(np.zeros(duration),color="tab:red",linestyle=(0, (5, 10)))
            solver_ax[-2].set_xlabel("Time s")
            solver_ax[-2].set_ylabel("Cost")

            solver_ax[-1].plot(computation_time, label="Computation Time (s)")
            solver_ax[-1].set_xlabel("Time s")
            solver_ax[-1].set_ylabel("Computation time")
            solver_ax[-1].grid(True)

            # save plot without mpc information
            for ax_ in solver_ax:
                ax_.legend(loc="upper left")
            solver_fig.tight_layout()
            solver_fig.savefig(experiment_path+"/plots/S"+str(exp_num)+"_NoPlanning_"+solver.name+"_trajectory")

            # Adding MPC information to the plots
            mpc_xs = solver.planning_state_history
            mpc_us = solver.planning_action_history
            #print(mpc_xs)
            for start_iteration, planned_states in mpc_xs:
                for i in range(state_size):
                    solver_ax[i].plot(range(start_iteration, start_iteration + planned_states.shape[0]),planned_states[:,i],alpha=0.4)#,label="$x_"+str(i)+"$",color="tab:orange")
            
            for start_iteration, planned_actions in mpc_us:
                for i in range(state_size,state_size+output_size):
                    if len(actions.shape) <=1:
                        actions = actions[:,None]
                    i_ = i-state_size
                    solver_ax[i].plot(range(start_iteration, start_iteration + planned_actions.shape[0]),planned_actions[:,i_],alpha=0.4)#, label="$u_"+str(i_)+"$",color="tab:green")

            # save plot with MPC information
            solver_fig.tight_layout()
            solver_fig.savefig(experiment_path+"/plots/S"+str(exp_num)+"_Planning_"+solver.name+"_trajectory")

            plt.close(solver_fig)
        env.history.to_csv(experiment_path+"/logs/S"+str(exp_num)+"_"+solver.name+"_log.txt")
        print(solver.name," finished Experiment No.",exp_num)

    if save_plots:
        for solver_ax in comb_ax:
            solver_ax.legend(loc="upper left")
            solver_ax.grid(True)
        figcomb.suptitle("Test No."+str(exp_num)+" | initial state "+str(reset_state))
        figcomb.tight_layout()
        figcomb.savefig(experiment_path+"/plots/CombS"+str(exp_num)+"_results")
        mpld3.save_html(figcomb,experiment_path+"/plots/CombS"+str(exp_num)+"_results.html")
        plt.close(figcomb)
