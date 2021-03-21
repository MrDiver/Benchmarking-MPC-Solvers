from MPCBenchmark.envs import Environment
from MPCBenchmark.models import Model
from MPCBenchmark.agents import Agent
import numpy as np
import time


class Experiment():
    def __init__(self, params: dict):
        """Experiment class to define a single rollout in a specific Environment

        Args:
            params (dict): 
            env : Environment (Class)
            model : Model (Class)
            agent : Agent (Class)
            agent_config : dict 
            experiment_length: int
            start_state: np.ndarray
        """
        self.Environment: Environment = params["env"]
        self.Model: Model = params["model"]
        self.Agent: Agent = params["agent"]
        self.agent_config: dict = params["agent_config"]
        self.experiment_length: int = params["experiment_length"]
        self.start_state: np.ndarray = params["start_state"]
        self.experiment_results = None
        # still not working correctly
        # self.goal_trajectory: np.ndarray = params["goal_trajectory"]

    def run(self, warmstart=None):
        """Runs the Experiment

        Returns:
            dict:
            "name": Experiment name (String)
            "computation_time": float
            "passed_time": float
            "env_states": ndarray
            "env_actions": ndarray
            "env_costs": ndarray
            "agent_planning_states": ndarray
            "agent_planning_actions": ndarray
            "agent_planning_costs": ndarray
        """
        # generate objects
        self.env = env = self.Environment()
        self.model = model = self.Model()
        agent = self.Agent(model, self.agent_config, cores=12)
        # goal_trajectory = np.zeros(
        #    (self.experiment_length+1, model.state_size+model.action_size))

        # reset
        env.reset(self.start_state)
        agent.reset()

        if warmstart is not None:
            agent.warm_start(env.state, warmstart, goal_state=np.zeros(model.state_size + model.action_size))

        starttime = time.time()
        passedtime = 0
        currenttime = time.time()
        computation_time = []
        # experiment loop
        for i in range(self.experiment_length):
            currenttime = time.time()
            action = agent.predict_action(
                env.state, goal_state=np.zeros(model.state_size + model.action_size))
            # newstate = model2.predict(env.state, action)
            # print(newstate)
            iterationtime = time.time() - currenttime
            passedtime += iterationtime
            currenttime += iterationtime
            iterationtime = np.around(iterationtime, decimals=3)
            computation_time.append(iterationtime)
            _, r, done, _ = env.step(action)

            print("==================")
            print("Solver:", agent.name)
            print("Config:", self.agent_config.get(
                "T"), self.agent_config.get("K"), self.agent_config.get("max_iter"))
            print("Time Passed:", passedtime)
            print("Iteration Time:", iterationtime)
            print("Cost", -r)

        passedtime = np.around(passedtime, decimals=3)
        states = np.array([x for x in env.history["state"].to_numpy()])
        actions = np.array([x for x in env.history["action"].to_numpy()])
        costs = np.array([x for x in env.history["cost"].to_numpy()])

        exp_name = agent.name + " solving " + env.name + " with " + model.name + \
                   "\n Starting from " + str(self.start_state) + \
                   " Time: " + str(passedtime)
        agent.close()
        self.experiment_results = {
            "name": exp_name,
            "computation_time": computation_time,
            "passed_time": passedtime,
            "warmstart": agent.warmstart,
            "warmstart_trajectories": np.concatenate(agent.warmstart_trajectories, axis=0),
            "env_name": env.name,
            "model_name": model.name,
            "agent_name": agent.name,
            "agent_config": self.agent_config,
            "env_start_state": self.start_state.tolist(),
            "env_states": states,
            "env_actions": actions,
            "env_costs": costs,
            "agent_planning_states": np.concatenate(agent.planning_state_history, axis=0),
            "agent_planning_actions": np.concatenate(agent.planning_action_history, axis=0),
            "agent_planning_costs": agent.planning_costs_history}

        return self.experiment_results

    def __call__(self, warmstart=None):
        """Runs the run Method of the experiment class

        Returns:
            dict:
            "name": Experiment name (String)
            "computation_time": float
            "passed_time": float
            "env_states": ndarray
            "env_actions": ndarray
            "env_costs": ndarray
            "agent_planning_states": ndarray
            "agent_planning_actions": ndarray
            "agent_planning_costs": ndarray
        """
        return self.run(warmstart=warmstart)
