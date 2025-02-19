import pandas as pd
from os import listdir
from os.path import isfile, join
import itertools
from yaml import safe_load


# Tools to summarize slurm results.

def load_experimental_grid(mypath,method_name="var_"):
    """
    Load all experiments in the specified path with the the specified method name.
    """
    experiments = [mypath + f for f in listdir(mypath) if not isfile(join(mypath, f))]
    experiments = [x for x in experiments if method_name in x]
    stack = []
    for experiment in experiments:
        res =load_experiment_results(experiment)
       
        stack.append(res)
    baseline_table = pd.concat(stack,axis=1)

    return baseline_table
    
    
def format_table(data,cfg, scoring= "AUROC", restriction=["normalize"]):
    
    data = data.drop(columns=[x for x in cfg.metrics if x != scoring])

    restriction = restriction + ["label_path"]
    grouped = data.groupby(restriction)
    options = list(itertools.product(*[data[x].unique() for x in restriction]))
    join = []
    for opt in options: 
        try:
            join.append(grouped.get_group(opt).sort_values(
                    scoring, ascending=False).iloc[0])
        except:
            print("Fail:",opt)
    best_runs =pd.concat(join,axis=1).T
    best_runs.T.loc[(best_runs.nunique() > 1).values].T
    return best_runs

def extract_raw_performance(formatted,cfg,scoring="AUROC",name="Combo", restriction=None):
    stack = []
    for x in cfg.ds_order:
        a = formatted[formatted["label_path"] == x]
        if restriction:
            a.index = ([name + "_" + str(y) for y in a[restriction]])
        else:
            a.index =  [name]
        a = a[[scoring]]
        a.columns = [x]
        stack.append(a)
    return pd.concat(stack,axis=1)


def load_experiment_results(experiment):
    stack = []#
    folders = [f for f in listdir(experiment)]
    for item in folders:
        # load performance
        performance = pd.read_csv(experiment + "/"+ item + "/" + "scoring.csv")
        performance.index = performance["Metric"]
        performance.drop(columns=["Metric"], inplace=True)    
        with open(experiment + "/"+ item + "/" + "config.yaml", 'r') as f:
            hps = pd.json_normalize(safe_load(f)).T
            hps.columns = performance.columns
        join = pd.concat([performance, hps],axis=0)
        join.loc["runtime", performance.columns] = pd.read_csv(experiment + "/"+ item + "/" + "runtime.csv").values[0,1]
        join.columns = [item]
        stack.append(join)
    return pd.concat(stack,axis=1)

    
#____________


def load_experiment_results_3(experiment):
    stack = []#
    namings =[]
    folders = [f for f in listdir(experiment)]
    scorings = []
    for item in folders:
        nameing = " ".join(item.split("_")[-3:])
        namings.append(nameing)
        # load performance
        res_data = [f for f in listdir(experiment + "/"+ item  )]

        for element in res_data: 
            if ".csv" in element:
                performance = pd.read_csv(experiment + "/"+ item + "/" + element)
                scorings.append(performance)
    return scorings,namings
