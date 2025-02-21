import hydra
from omegaconf import DictConfig
import os
from assembly_tools import load_experimental_grid

# This script was used to extract the results of the grid search.


@hydra.main(version_base=None, config_path="config", config_name="extract.yaml")
def main(cfg: DictConfig):
    print(cfg)

    data = load_experimental_grid(mypath=cfg.data_path + cfg.exp, method_name=cfg.method_name).T

    # test for consistency
    # Fix old hydra specs. Does nothing if strictly  version is used.
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

    print("Number of samples in grid:", len(data))
    # Issue with list specification
    data["data_preprocess.subset_month"] = data["data_preprocess.subset_month"].astype(str)
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
    if cfg.exp == "exp2/":
        relevant_hps = relevant_hps + ["data_path", "data_preprocess.subset_year", "data_preprocess.subset_month"]
    
    # label path parsing
    str_data_path = data["label_path"].str.contains("/")
    data.loc[str_data_path, "label_path"] = [
        x[3] for x in data[str_data_path]["label_path"].str.split("/").values
    ]
    data = data[cfg.metrics + relevant_hps]

    print("Dropping duplicates:")
    print(len(data))
    #Runtime will never match...
    data = data.drop_duplicates(subset=[x for x in list(data.columns) if x != "runtime"])
    print(len(data))

    assert int(data[relevant_hps].duplicated().sum()) == 0, "Duplicated columns!!"

    if not os.path.exists(cfg.save_path + cfg.exp):
        os.makedirs(cfg.save_path+ cfg.exp)
    data.to_csv(cfg.save_path + cfg.exp + "/" + cfg.method_name + ".csv")


if __name__ == "__main__":
    main()
