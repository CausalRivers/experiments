
import numpy as np
import mxnet as mx
from methods.deep_ar_tools import train_deep_ar_estimator as train_estimator
from methods.deep_ar_tools import eval_deep_ar_estimator as eval_func
from methods.cdmi_tools import cdmi

# Adapted from CDMI_light repository.
def cdmi_baseline(data_sample, cfg):
    np.random.seed(1)
    mx.random.seed(2)

    training_data = data_sample.iloc[:cfg.training_length]
    training_length = cfg.training_length
    num_windows = cfg.num_windows

    M = train_estimator(training_data,cfg) 
    # careful! depending on the metric smaller is more causal (p-value)
    pvals_stack  = cdmi(
        data_sample, M,eval_func,training_length, num_windows, cfg
    )
    res = 1- pvals_stack # to map this properly to p-values.

    # remove batch dimension here.
    if np.isnan(np.array(res)).sum() > 0:
        print("ISSUE detected, prediction set to 0") 
        res = np.zeros(res.shape)
    return res




