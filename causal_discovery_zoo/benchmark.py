import sys
import pickle
import numpy as np
import hydra
from omegaconf import DictConfig
from tools.tools import benchmarking, score
import pandas as pd
import os
import pickle
from omegaconf import OmegaConf
import datetime

from tools.tools import (
    summary_transform,
    load_single_samples,
    load_joint_samples,
    benchmarking,
    datetime_preprocessing,
)


# Example script to benchmark causal discovery methods.
@hydra.main(version_base=None, config_path="config", config_name="benchmark.yaml")
def main(cfg: DictConfig):

    start = datetime.datetime.now()
    print(cfg)
    # handle different envs as all of these libraries are poorely maintained we need multiple envs.
    if cfg.method.name == "reverse_physical":
        from methods.baseline_methods import remove_edges_via_mean_values as cd_method
    elif cfg.method.name == "cross_correlation":
        from methods.baseline_methods import (
            cross_correlation_for_causal_discovery as cd_method,
        )
    elif cfg.method.name == "combo":
        from methods.baseline_methods import combo_baseline as cd_method
    elif cfg.method.name == "var":
        from methods.var import var_baseline as cd_method
    elif cfg.method.name == "pcmci":
        from methods.pcmci import pcmci_baseline as cd_method
    elif cfg.method.name == "dynotears":
        from methods.dynotears import dynotears_baseline as cd_method
    elif cfg.method.name == "varlingam":
        from methods.varlingam import varlingam_baseline as cd_method
    elif cfg.method.name == "cdmi":
        from methods.cdmi import cdmi_baseline as cd_method
    elif cfg.method.name == "cp":
        from methods.causal_pretraining import causal_pretraining_baseline as cd_method
    elif cfg.method.name == "stic":
        from methods.stic import stic_baseline as cd_method
    elif cfg.method.name == "tcdf":
        from methods.tcdf import tcdf_baseline as cd_method
    else:
        raise ValueError("Invalid method")

    if cfg.label_path[-1] == "/":
        print("Folder specified.Attempting single load.")
        test_data, test_labels = load_single_samples(cfg)
    else:
        print("File specified. Attempting joint load.")
        test_data, test_labels = load_joint_samples(
            cfg, index_col=cfg.index_col, preprocessing=datetime_preprocessing if cfg.dt_preprocess else None
        )
    preds = benchmarking(test_data, cfg, cd_method)

    if test_labels[0].ndim == 2 and preds[0].ndim == 3:
        # reduce lag dimension according so config
        preds = [summary_transform(x, cfg) for x in preds]

    preds = np.array(preds)
    test_labels = np.array(test_labels)
    out = score(preds, test_labels, cfg)
    print(out)
    if cfg.save_full_out:

        # make folder with naming
        p = cfg.save_path + cfg.method.name + "_" + cfg.label_path.split("/")[-2]
        if not os.path.exists(p):
            os.makedirs(p)
        inner_p = p + "/" + str(datetime.datetime.now())[:24]
        os.makedirs(inner_p)
        out.to_csv(inner_p + "/scoring.csv")
        stop_time = datetime.datetime.now() - start
        pd.DataFrame([stop_time], columns=["runtime"]).to_csv(
            inner_p + "/runtime.csv"
        )  # dumps to file:
        with open(inner_p + "/config.yaml", "w") as f:
            OmegaConf.save(cfg, f)
        pickle.dump(preds, open(inner_p + "/preds.p", "wb"))
    print("Done", stop_time)


if __name__ == "__main__":
    main()
