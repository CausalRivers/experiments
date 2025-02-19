import sys
import pickle
import numpy as np
import hydra
from omegaconf import DictConfig
import sys
from tools.tools import benchmarking, score, preprocessing_pipeline
import pandas as pd
import os
import pickle
from omegaconf import OmegaConf
import datetime
from os import listdir
from additional_methods.causal_pretraining import causal_pretraining_baseline  as baseline_func
from additional_methods.causal_pretraining import Architecture_PL
from os.path import isfile, join

# Example script to benchmark the baseline methods.

@hydra.main(version_base=None, config_path="config", config_name="eval_finetune.yaml")
def main(cfg: DictConfig):

    
    print(cfg)
    start = datetime.datetime.now()
    print("Loading data...")


    from os import listdir
    from os.path import isfile, join
    mypath = cfg.finetune_model_path
    all_models = sorted([f for f in listdir(mypath) if not isfile(join(mypath, f))])
    print(all_models)
    selection = all_models[cfg.selected_model]
    print(selection)
    p = mypath + "/" + selection
    finetune_paths  = sorted([p+ "/" + f for f in listdir(
        p) if isfile(join(p, f))])

    finetune_paths = [x for x in finetune_paths if not "config" in x]
    test_data = pickle.load(open(cfg.sample_path + cfg.ds_name + "/test.p", "rb"))

    if cfg.restrict_to:
        test_data = test_data[:cfg.restrict_to]

    Y,Y_names,X = preprocessing_pipeline(test_data,cfg)
    naming = str(cfg.normalize) + "_" + str(cfg.resolution_value) + "_" + str(cfg.map_to_summary_graph)

    outs = []
    for path in finetune_paths:
        preds = baseline_func(X,Y_names,cfg, provided_model=path, adapted_architecture=True)
        outs.append(score(preds,Y,cfg, provided_name=path + "_" + naming))



    outs = pd.concat(outs, axis=1)
    print(outs)
    inner_p = cfg.save_path  + "/" +  selection + "_" + cfg.ds_name + "/"
    if not os.path.exists(inner_p):
        os.makedirs(inner_p)

    naming = str(cfg.normalize) + "_" + str(cfg.resolution_value) + "_" + str(cfg.map_to_summary_graph)

    outs.to_csv(inner_p + "/" + naming +".csv")
    with open(inner_p + "/" + naming + ".yaml", "w") as f:
        OmegaConf.save(cfg, f)


if __name__ == "__main__":
    main()