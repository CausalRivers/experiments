import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    accuracy_score,
)

# precision recall warning.
np.seterr(divide="ignore", invalid="ignore")





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


def score(preds, labs, remove_autoregressive=True, name="Result"):
    """
    Calculates a number of metrics given preds and labs.
    Takes in either a 2dim or a 3dim tensor (batch of summary graphs)
    name is used for later column naming.
    """
    
    # Some casting concerning input data type:
    if isinstance(preds, list):
        preds = np.array(preds) 
    if isinstance(labs, list):
        labs = np.array(labs) 
    if isinstance(preds,pd.DataFrame):
        preds = preds.values
    if isinstance(labs,pd.DataFrame):
        labs = labs.values
    # expand dims if a single samle is provided
    if preds.ndim == 2:
        preds = np.expand_dims(preds, 0)
        labs = np.expand_dims(labs, 0)

    # Calculates a number of metrics and returns a df holding them
    print("Scoring...")
    # We remove the diagonal as rivers are highly autocorrelated and the causal links here are not relevant.
    if remove_autoregressive:
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
        # columns=[cfg.method.name + "_" + cfg.label_path.split("/")[-2]],
        columns=[name],
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
