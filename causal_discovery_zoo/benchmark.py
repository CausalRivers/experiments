import hydra
from omegaconf import DictConfig
import datetime

from tools.tools import (
    load_joint_samples,
    benchmarking,
    standard_preprocessing,
    save_run,
    summary_transform,
)
from tools.scoring_tools import score


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
    else:
        raise ValueError("Invalid method")

    print("File specified. Attempting joint load.")
    test_data, test_labels = load_joint_samples(
        cfg, preprocessing=standard_preprocessing if cfg.dt_preprocess else None
    )
    preds = benchmarking(test_data, cfg, cd_method)
    if test_labels[0].ndim == 2 and preds[0].ndim == 3:
        # reduce lag dimension according so config
        preds = [summary_transform(x, cfg.map_to_summary_graph) for x in preds]


    print(preds[0])
    out = score(preds, test_labels, cfg.remove_diagonal, name=cfg.method.name)

    stop_time = datetime.datetime.now() - start
    print(out)
    if cfg.save_full_out:
        print("Saving...")
        save_run(out, stop_time, preds, cfg)
    print("Done", stop_time)


if __name__ == "__main__":
    main()
