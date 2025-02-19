import hydra
from omegaconf import DictConfig
import os
from assembly_tools import load_experimental_grid

# This script was used to extract the results of the grid search.


@hydra.main(version_base=None, config_path="config", config_name="extract.yaml")
def main(cfg: DictConfig):
    print(cfg)

    data = load_experimental_grid(mypath=cfg.data_path, method_name=cfg.method_name).T

    # test for consistency
    # Fix old hydra specs. Does nothing if strictly  version is used.
    for x in cfg.rename:
        if cfg.rename[x] in data.columns:
            data.loc[data[cfg.rename[x]].isnull(), cfg.rename[x]] = data.loc[
                data[cfg.rename[x]].isnull(), x
            ]
        data.drop(columns=x, inplace=True)
    print("Number of samples in grid:", len(data))

    # TODO Fix when relevant for exp2
    data["window_data_month_value"] = data["window_data_month_value"].astype(str)
    # nothing should vary outside of the hps and the metrics.
    control = data[
        [x for x in data.columns if ((x not in cfg.metrics) and (x not in cfg.hp_list))]
    ]
    control = control.T[control.nunique() > 1]

    # Whatever remains here should come from inconsistencies.
    print("Validate! The following additional columns have non unique values:")
    for x in control.T.columns:
        print(control.T[x].value_counts())

    relevant_hps = [x for x in cfg.hp_list if x in data.columns]
    # label path parsing
    str_data_path = data["label_path"].str.contains("/")
    data.loc[str_data_path, "label_path"] = [
        x[3] for x in data[str_data_path]["label_path"].str.split("/").values
    ]
    data = data[cfg.metrics + [x for x in cfg.hp_list if x in data.columns]]

    assert int(data[relevant_hps].duplicated().sum()) == 0, "Duplicated columns!!"

    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
    data.to_csv(cfg.save_path + "/" + cfg.method_name + ".csv")


if __name__ == "__main__":
    main()
