import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    accuracy_score,
)
from os import listdir
from os.path import isfile, join
import pickle

# To prevent F1 max spam due to 0 divide
np.seterr(divide='ignore', invalid='ignore')


def make_human_readable(out, d):
    """
    Transforms a 
    """
    out = pd.DataFrame(out, columns=d.columns, index=d.columns)
    out = pd.concat([pd.concat([out], keys=["Cause"], axis=1)], keys=["Effect"])
    return out


def remove_trailing_nans(sample_prep):
    """
    Removes samples that were not removed by interpolate.
    """

    check_trailing_nans = np.where(sample_prep.isnull().values.any(axis=1) == 0)[0]
    if not len(check_trailing_nans) == 0:  # A ts is completely 0:
        sample_prep = sample_prep[
            check_trailing_nans.min() : check_trailing_nans.max() + 1
        ]
    else:
        sample_prep = sample_prep.fillna(value=0)
    assert sample_prep.isnull().sum().max() == 0, "Check nans!"

    return sample_prep


def datetime_preprocessing(
    data : pd.DataFrame,
    cfg,
):
    # Adjust resolution
    sample_data = data.copy()  # dont change the original data
    sample_data["dt"] = pd.to_datetime(data.index).round(cfg.resolution).values

    sample_data = sample_data.groupby("dt").mean()
    # subsampling
    if cfg.subset:
        sample_data = sample_data.loc[
            (sample_data.index.month.isin(cfg.subset[1]))
            & (sample_data.index.year == cfg.subset[0])
        ]
    sample_data = sample_data.iloc[::cfg.subsample, :]
    if cfg.normalize:
        sample_data = (sample_data - sample_data.min()) / (
            sample_data.max() - sample_data.min()
        )
    if cfg.interpolate:
        sample_data = sample_data.interpolate()
    return sample_data


def benchmarking(X, cfg, method_to_test):
    """
    Takes in the output of the data loader and perform the predictions with a specified method.
    If anything else should happen with the data beforehand we should perform this here.
    """
    preds = []
    for x, sample in enumerate(X):
        inp = remove_trailing_nans(sample)
        print(x, "/", len(X))
        preds.append(method_to_test(inp, cfg.method))
    return preds


def load_single_samples(cfg):
    # Load data. replace this here if necessary.
    onlyfiles = [
        f
        for f in listdir(cfg.label_path)
        if isfile(join(cfg.label_path, f))
    ]
    test_data = sorted([x for x in onlyfiles if "data" in x])
    test_labels = sorted([x for x in onlyfiles if "label" in x])

    if cfg.restrict_to >= 0:
        test_data = test_data[cfg.restrict_to : cfg.restrict_to + 1]
        test_labels = test_labels[cfg.restrict_to : cfg.restrict_to + 1]

    test_data = [
        pd.read_csv(cfg.label_path + sample, index_col=0)
        for sample in test_data
    ]
    test_labels = [
        pd.read_csv(cfg.label_path + sample, index_col=0)
        .astype(bool)
        .values
        for sample in test_labels
    ]

    return test_data, test_labels


def load_joint_samples(cfg, index_col="datetime", preprocessing=None):
    """
    Loads and transforms the data.
    if you have an index_col= specify it.
    If you have additional preprocessing you can provide a function.
    """

    data = pickle.load(open(cfg.label_path, "rb"))
    if cfg.restrict_to >= 0:
        data = data[cfg.restrict_to : cfg.restrict_to + 1]
    # This is not ram efficient but faster to process.
    Y = [graph_to_label_tensor(sample, human_readable=True) for sample in data]
    # To fix double col names due to human readable format.
    Y_names = [[m[1] for m in sample.columns.values] for sample in Y]
    unique_nodes = list(set([item for sublist in Y_names for item in sublist]))
    unique_nodes = (
        ([index_col] + [str(x) for x in unique_nodes])
        if index_col
        else [str(x) for x in unique_nodes]
    )
    data = pd.read_csv(cfg.data_path,
        index_col=index_col if index_col else None,
        usecols=unique_nodes,
    )
    if preprocessing:
        data = preprocessing(data, cfg.data_preprocess)
    X = []
    for sample in Y:
        X.append(data[[str(m[1]) for m in sample.columns]])
    return X, Y


def summary_transform(pred, cfg):
    if cfg.map_to_summary_graph == "max":
        prediction = pred.max(axis=2)
    elif cfg.map_to_summary_graph == "mean":
        prediction = pred.mean(axis=2)
    return prediction


def remove_diagonal(T):
    # Takes in 3 dim tensor and removes diagonal of 2/3 dim.
    out = []
    for x in T:
        out.append(x[~np.eye(x.shape[0], dtype=bool)].reshape(x.shape[0], -1))
    return np.stack(out)


def max_accuracy(labs, preds):
    # ACCURACY MAX
    preds = preds.astype(float)
    if preds.min() == preds.max():
        a = []
    else:
        a = list(
            np.arange(
                preds.min(),
                preds.max() + preds.min(),
                (preds.max() - preds.min()) / 100,
            )
        )  # 100 steps
    possible_thresholds = [0] + a + [preds.max() + 1e-6]
    acc = [accuracy_score(labs, preds > thresh) for thresh in possible_thresholds]
    acc_thresh = possible_thresholds[np.argmax(acc)]
    acc_score = np.nanmax(acc)
    return acc_thresh, acc_score


def f1_max(labs, preds):
    # F1 MAX
    precision, recall, thresholds = precision_recall_curve(labs, preds)
    f1_scores = 2 * recall * precision / (recall + precision)
    f1_thresh = thresholds[np.argmax(f1_scores)]
    f1_score = np.nanmax(f1_scores)
    return f1_thresh, f1_score


def score(preds, labs, cfg):

    # Calculates a number of metrics and returns a df holding them
    print("Scoring...")
    # We remove the diagonal as rivers are highly autocorrelated and the causal links here are not relevant.
    if cfg.remove_diagonal:
        labs = remove_diagonal(labs)
        preds = remove_diagonal(preds)
    else:
        labs = np.array(labs)
        preds = np.array(preds)
    # Individual scoring for each sample.

    f1_max_ind = []
    f1_thresh_ind = []
    accuracy_ind = []
    accuracy_ind_thresh = []
    auroc_ind = []
    for x in range(len(labs)):
        print(x, "/", len(labs))
        if len(set(labs[x].flatten())) == 1:
            # not defined for empty samples
            # this can sometimes happen if a limited time window is chosen
            continue
        else:
            auroc_ind.append(
                roc_auc_score(y_true=labs[x].flatten(), y_score=preds[x].flatten())
            )
        f1_thresh, f1_score = f1_max(labs[x].flatten(), preds[x].flatten())
        f1_max_ind.append(f1_score)

        f1_thresh_ind.append(f1_thresh)
        acc_thresh, acc_score = max_accuracy(labs[x].flatten(), preds[x].flatten())
        accuracy_ind.append(acc_score)
        accuracy_ind_thresh.append(acc_thresh)

    f1_max_ind = np.array(f1_max_ind).mean()
    f1_thresh_ind = np.array(f1_thresh_ind).mean()
    accuracy_ind = np.array(accuracy_ind).mean()
    accuracy_ind_thresh = np.array(accuracy_ind_thresh).mean()
    auroc_ind = np.array(auroc_ind).mean()

    # Joint calculation
    labs = labs.flatten()
    preds = preds.flatten()

    # AUROC
    auroc = roc_auc_score(labs, preds)
    # F1 MAX

    f1_thresh, f1_score = f1_max(labs, preds)
    # ACCURACY MAX
    acc_thresh, acc_score = max_accuracy(labs, preds)

    null_model_auroc = roc_auc_score(labs, np.zeros(preds.shape))

    _, null_model_f1 = f1_max(labs, np.zeros(preds.shape))

    _, null_model_acc = max_accuracy(labs, np.zeros(preds.shape))

    out = pd.DataFrame(
        [
            acc_thresh,
            acc_score,
            accuracy_ind_thresh,
            accuracy_ind,
            null_model_acc,
            f1_thresh,
            f1_score,
            f1_thresh_ind,
            f1_max_ind,
            null_model_f1,
            auroc,
            auroc_ind,
            null_model_auroc,
        ],
        columns=[cfg.method.name + "_" + cfg.label_path.split("/")[-2]],
        index=[
            "Max Acc thresh",
            "Max Acc",
            "Max individual Acc thresh",
            "Max individual Acc",
            "Null Acc",
            "Max F1 thresh",
            "Max F1",
            "Max individual F1 thresh",
            "Max individual F1",
            "Null F1",
            "AUROC",
            "Individual AUROC",
            "Null AUROC",
        ],
    )
    out.index.name = "Metric"
    return out


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
