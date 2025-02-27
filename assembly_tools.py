import pandas as pd
from os import listdir
from os.path import isfile, join
import itertools
from yaml import safe_load
import numpy as np
from hydra import initialize, compose

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
            a.index = ([name + "_" + str(y) for y in a[restriction].astype(str).agg("-".join,axis=1)])
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

    
def load_finetuning(path = "causal_discovery_zoo/cp_finetuned/"):
    stack = []
    onlyfiles = [path + f for f in listdir(path)]
    
    for run in onlyfiles:
        try:
            test = pd.read_csv(run + "/test_performance.csv", index_col=0)
            val = pd.read_csv(run + "/val_performance.csv", index_col=0)
            with open(run  + "/config.yaml", 'r') as f:
                hps = pd.json_normalize(safe_load(f))
            info = pd.concat([hps,val,test],axis=1).T
            info.columns = [run.split("/")[-1]]
            stack.append(info)
        except:
            print("fail:", run)
    return pd.concat(stack,axis=1).T




def build_exp1(scoring="Max individual F1", null_name="Null F1"):

    with initialize(version_base=None, config_path="config/"):
        cfg = compose(config_name='extract.yaml')
    hp_stack = []
    performance_stack = []

    for name in ["reverse", "cross", "combo", ]:
        data = pd.read_csv("grid_export_exp1/" + name + ".csv", index_col=0)
        # quick fix
        res = "method.filter_mode"

        data.loc[data[res].isnull(),res] = "none"
        formatted = format_table(data,cfg, scoring= scoring, restriction=[res] if res else [])
        lines = extract_raw_performance(formatted,cfg,scoring,name=name, restriction=[res])
        hp_stack.append(formatted)
        performance_stack.append(lines)
        
    for name in ["var_", "varlingam", "dynotears", "pcmci"]:
        data = pd.read_csv("grid_export_exp1/" + name + ".csv", index_col=0)
        # quick fix
        res = None
        formatted = format_table(data,cfg, scoring= scoring, restriction=[res] if res else [])
        lines = extract_raw_performance(formatted,cfg,scoring,name=name, restriction=None)
        hp_stack.append(formatted)
        performance_stack.append(lines)
        
    bs = data[[null_name, "label_path"]].groupby("label_path").max().T
    bs.index = ["Null model"]

    # load this direct as we ran this completely seperated.
    #data = load_experimental_grid(mypath=cfg.data_path + "exp1/", method_name="cdmi").T
    # data.to_csv(cfg.data_path + "exp1/cached.csv")
    data = pd.read_csv(cfg.data_path + "exp1/cached.csv", index_col=0)
    for x in cfg.rename:
        if x in data.columns:
            print("renaming..", x)
            if cfg.rename[x] in data.columns:
                data.loc[data[cfg.rename[x]].isnull(), cfg.rename[x]] = data.loc[
                    data[cfg.rename[x]].isnull(), x
                ]
                data.drop(columns=x, inplace=True)
            else:
                data.rename(columns={x: cfg.rename[x]}, inplace=True)
    data = data[[scoring, "label_path"]].groupby("label_path").mean()
    data = data.rename(index= {"../../datasets/close_3/east.p": "close_3","../../datasets/close_5/east.p":"close_5"}).loc[cfg.ds_order].T
    data.index.name = None
    data.index = ["CDMI"]
    performance_stack.append(data)

    for name in ["cp"]:
        data = pd.read_csv("grid_export_exp1/" + name + ".csv", index_col=0)
        # quick fix
        res = "method.architecture"
        cfg.ds_order = cfg.ds_order[:-1]
        formatted = format_table(data,cfg, scoring= scoring, restriction=[res] if res else [])
        lines = extract_raw_performance(formatted,cfg,scoring,name=name, restriction=[res])
        hp_stack.append(formatted)
        performance_stack.append(lines)
        
    performance_stack.append(bs)
    out = pd.concat(performance_stack)
    
    out = out.loc[out.index[[16,1,2,0,3,4,8,6,7,5,9,10,11,12,13,14,15]]]

    out.rename(index={
    "cross_none": "CC",
    "cross_corr": "CC+C",
    "reverse_none": "RP",
    "reverse_biggest": "RP+B",
    "reverse_next": "RP+N",
    "combo_none": "RPCC",
    "combo_biggest": "RPCC+B",
    "combo_next": "RPCC+N",
    "combo_corr": "RPCC+C",
    "cp_unidirectional": "CP (Gru)",
    "cp_transformer":"CP (Transf)",
    "var_": "VAR",
    "varlingam": "Varlingam",
    "pcmci": "PCMCI"
    }, inplace=True)

    out = out.astype(float)

    def highlight_max(s, props=''):
        return np.where(s == np.nanmax(s.values), props, '')


    def highlight_second_min(s, props=''):
        return np.where(s == s.iloc[1:].min(), props, '')

    def highlight_second_max(s, props=''):
        return np.where(s == s.nlargest(2).values[1], props, '')


    c1 = 'background-color:#b8f2be'
    c2 = 'background-color:#a5cfa9'
    c3 = 'background-color:#e8af97'

    styling = out.style.apply(highlight_max, props=c1, axis=0).apply(highlight_second_max, props=c2, axis=0).format(precision=2)

    print(styling.format(precision=2).to_latex(convert_css=True).replace("0.", ".").replace("nan", "\\textdagger").replace(" .", "."))
    return out