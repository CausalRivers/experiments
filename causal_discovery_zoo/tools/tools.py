import numpy as np
import pandas as pd
import pickle
import os
import datetime
from omegaconf import OmegaConf


def summary_transform(pred, opt):
    if opt == "max":
        prediction = pred.max(axis=2)
    elif opt == "mean":
        prediction = pred.mean(axis=2)
    return prediction


def remove_trailing_nans(sample_prep):
    """
    Removes samples that were not removed by interpolate.
    """
    check_trailing_nans = np.where(sample_prep.isnull().values.any(axis=1) == 0)[0]
    if not len(check_trailing_nans) == 0:  # A ts is completely 0:
        sample_prep = sample_prep[
            check_trailing_nans.min() : check_trailing_nans.max() + 1
        ]
        
    if len(sample_prep) == 0: 
        # random case that everything is empty. This can happen when selecting a window.
        print("EMPTY SAMPLE DETECTED")
    return sample_prep

# Subsample only all subgraphs that contain only saxony and thuringia nodes:
def filter_samples_based_on_properties(ds, G, selection=["T", "S"], prop="origin"):
    sub_ds = []
    for d in ds:
        origin_check = set([G.nodes[x][prop] for x in d.nodes])
        if np.all([x in selection for x in origin_check]):
            sub_ds.append(d)
    return sub_ds


def graph_to_label_tensor(G_sample, human_readable=False):
    nodes = sorted(G_sample.nodes)
    labels = np.zeros((len(nodes), len(nodes)))

    for n, x in enumerate(nodes):
        for m, y in enumerate(nodes):
            if (x, y) in G_sample.edges:
                labels[m, n] = 1
    if human_readable:
        labels = pd.DataFrame(labels, columns=nodes, index=nodes)
        labels = pd.concat(
            [pd.concat([labels], keys=["Cause"], axis=1)], keys=["Effect"]
        )
        return labels
    else:
        return labels


def load_sample(which, p="resources/rivers_ts_east_germany.csv"):
    return pd.read_csv(
        p, index_col=0, usecols=["datetime"] + [str(x) for x in list(which.nodes)]
    )



def preprocess_data(
    data,
    resolution="2H",
    interpolate=True,
    subset_year=False,
    subset_month=False,
    subsample=1,
    normalize=False,
    remove_trailing_nans_early=False,
):

    sample_data = data.copy()  # dont change the original data

    # Remove trailing nans (so start and end of the ts to improve data quality.
    # WARNING: This can make the TS arbitrarily short).
    if remove_trailing_nans_early:
        sample_data = remove_trailing_nans(sample_data)
        
    # Adjust resolution
    sample_data["dt"] = pd.to_datetime(sample_data.index).round(resolution).values

    sample_data = sample_data.groupby("dt").mean()
    # subsampling
    if subset_month:
        sample_data = sample_data.loc[
            (sample_data.index.month.isin(subset_month))
            & (sample_data.index.year == subset_year)
        ]
    sample_data = sample_data.iloc[::subsample, :]
    if normalize:
        sample_data = (sample_data - sample_data.min()) / (
            sample_data.max() - sample_data.min()
        )
    if interpolate:
        sample_data = sample_data.interpolate()
    return sample_data


def standard_preprocessing(
    data: pd.DataFrame,
    cfg,
):
    """
    simple wrapper arround the standard preprocessing class that uses a hydra config only.
    """
    sample_data = preprocess_data(
        data,
        resolution=cfg.resolution,
        interpolate=cfg.interpolate,
        subset_year=cfg.subset_year,
        subset_month=cfg.subset_year,
        subsample=cfg.subsample,
        normalize=cfg.normalize,
        remove_trailing_nans_early=cfg.remove_trailing_nans_early
    )
    return sample_data


def benchmarking(X, cfg, method_to_test):
    """
    Takes in the output of the data loader and perform the predictions with a specified method.
    If anything else should happen with the data beforehand this should happen here.
    """
    preds = []
    for x, sample in enumerate(X):
        print(x, "/", len(X))
        preds.append(method_to_test(sample, cfg.method))
    return preds


def load_joint_samples(cfg, index_col="datetime", preprocessing=None):
    """
    Loads and transforms the data.
    If you have additional preprocessing you can provide a function.
    Importantly, if you struggle with ram it migt be worth to load the samples individually as in 2_tutorial_benchmarking.
    This is however slower.
    """

    data = pickle.load(open(cfg.label_path, "rb"))
    # restrict which unique sample you want to process
    if cfg.restrict_to >= 0:
        data = data[cfg.restrict_to : cfg.restrict_to + 1]
    # This is not ram efficient but faster to process.
    Y = [graph_to_label_tensor(sample, human_readable=True) for sample in data]
    # To fix double col names due to human readable format.
    Y_names = [[m[1] for m in sample.columns.values] for sample in Y]
    # Get all required ts
    unique_nodes = list(set([item for sublist in Y_names for item in sublist]))
    unique_nodes = (
        ([index_col] + [str(x) for x in unique_nodes])
        if index_col
        else [str(x) for x in unique_nodes]
    )
    # load required files
    data = pd.read_csv(
        cfg.data_path,
        index_col=index_col if index_col else None,
        usecols=unique_nodes,
    )
    # apply specify preprocessing to the data
    if preprocessing:
        data = preprocessing(data, cfg.data_preprocess)
    # Again, this is not Ram efficient but it loads all the samples jointly.
    X = []
    for sample in Y:
        single_sample = data[[str(m[1]) for m in sample.columns]]
        # final nan removal if anyything remains.
        single_sample = remove_trailing_nans(single_sample)
        X.append(single_sample)
        
        # PUT IN REMOVE TRAILING NANS HERE AND USE IT earlier also.
    return X, Y


def save_run(out,stop_time, preds, cfg):
    # make folder with naming
    p = cfg.save_path + cfg.method.name + "_" + cfg.label_path.split("/")[-2]
    if not os.path.exists(p):
        os.makedirs(p)
    inner_p = p + "/" + str(datetime.datetime.now())[:24]
    os.makedirs(inner_p)
    out.to_csv(inner_p + "/scoring.csv")

    pd.DataFrame([stop_time], columns=["runtime"]).to_csv(
        inner_p + "/runtime.csv"
    )  # dumps to file:
    with open(inner_p + "/config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    pickle.dump(preds, open(inner_p + "/preds.p", "wb"))
