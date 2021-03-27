from MPCBenchmark.agents import Agent, CEM, MPPI, ILQR
from MPCBenchmark.envs import Environment, PendulumEnv, CartPoleSwingUpEnv, AcrobotEnv
from MPCBenchmark.models import Model, PendulumModel, CartPoleSwingUpModel, AcrobotModel
from MPCBenchmark.ExperimentCore import Experiment, Plot, DBTools
import numpy as np
import pandas as pd
from pymongo import MongoClient
import os
import pprint


def highlight_max_value(series):    # get True, or False status of each value in series
    boolean_mask = series == series.max()    # return color is orange when the boolean mask is True
    res = [f"color : orange" if max_val else '' for max_val in boolean_mask]
    return res



def generate_tables():
    client = MongoClient("192.168.0.101", 27017)
    db = client.parameter_tuning
    collections = [db.cem_ratios, db.ilqr_runs2, db.mppi_samples, db.temperature_exp]
    pd.set_option('display.float_format', '{:.2g}'.format)

    names = {"avg": "Average", "min": "Min", "med": "Median", "q25": "Q 25", "q75": "Q 75",
             "time": "Time"}

    def query_data(query):
        times = []
        costs = []
        #print(query)
        for collection in collections:
            for result in collection.find(query):
                tmp = DBTools.decodeDict(result)
                times.append(tmp["passed_time"])
                costs.append(tmp["env_costs"])
        costs = np.clip(costs, -20, 20)
        avg = np.mean(np.sum(costs, axis=1))
        min = np.min(np.sum(costs, axis=1))
        median = np.median(np.sum(costs, axis=1))
        q25th = np.quantile(np.sum(costs, axis=1), 0.25)
        q75th = np.quantile(np.sum(costs, axis=1), 0.75)
        time = np.sum(times)
        result = {names["avg"]: avg, names["min"]: min, names["med"]: median, names["q25"]: q25th, names["q75"]: q75th,
             names["time"]: time}
        return result

    def format_table(table):
        res = table.min()
        res = res.apply((lambda x: int(np.floor(np.log10(x)))))
        powers = res.apply((lambda x: np.power(10, x)))

        def transform(x):
            return x / powers

        new_table = table.apply(transform, axis=1)

        rename_query = {i: i + r" $(1 \times 10^" + str(res[i]) + ") $" for i in table.columns}
        new_table.rename(columns=rename_query, inplace=True)
        new_table = new_table.to_latex(escape=False, decimal=",")
        return new_table

    def write_latex_table(path, table):
        f = open(path, "w")
        lines = [r"\documentclass[]{standalone}",
                 "\n",
                 r"\usepackage{booktabs}",
                 "\n",
                 r"\usepackage[table,xcdraw]{xcolor}",
                 "\n",
                 r"\begin{document}",
                 "\n",
                 table,
                 "\n",
                 r"\end{document}"]
        f.writelines(lines)
        f.close()

    #Generate Table part
    for env, statesize in [("PendulumEnvironment", 2), ("CartpoleSwingupEnvironment", 4), ("AcrobotEnvironment", 4)]:
        if not os.path.exists("paper"):
            os.mkdir("paper")
        if not os.path.exists("paper/time_comparison"):
            os.mkdir("paper/time_comparison")

        # Generate Time Horizon Tables
        df = pd.DataFrame()
        for i, T in enumerate([5, 10, 25, 50]):
            for solver in ["CEM", "MPPI", "ILQR"]:
                query = {"env_name": env, "agent_name": solver, "agent_config.T": T}
                res = query_data(query)
                res["T"] = T
                res["Solver"] = solver
                df = df.append(res, ignore_index = True)
                df = df[["T","Solver",names["avg"],names["min"], names["med"],names["q25"],names["q75"],names["time"]]]
        sorted_table = pd.pivot_table(df,[names["avg"],names["min"], names["med"],names["q25"],names["q75"],names["time"]],["T","Solver"])

        latex_format =  format_table(sorted_table)
        write_latex_table("paper/time_comparison/"+env+"_T_table.tex",latex_format)

        # Generate Sample Tables
        df = pd.DataFrame()
        K = [10, 20, 50, 200 , 500]
        for i, K in enumerate(K):
            for solver in ["CEM", "MPPI"]:
                if solver == "CEM" and K == 500:
                    continue
                query = {"env_name": env, "agent_name": solver, "agent_config.K": K}
                res = query_data(query)
                res["K"] = K
                res["Solver"] = solver
                df = df.append(res, ignore_index=True)
                df = df[["K", "Solver", names["avg"], names["min"], names["med"], names["q25"], names["q75"],
                         names["time"]]]
            sorted_table = pd.pivot_table(df, [names["avg"], names["min"], names["med"], names["q25"], names["q75"],
                                               names["time"]], ["K", "Solver"])

        latex_format = format_table(sorted_table)
        write_latex_table("paper/time_comparison/" + env + "_K_table.tex", latex_format)


if __name__ == '__main__':
    generate_tables()