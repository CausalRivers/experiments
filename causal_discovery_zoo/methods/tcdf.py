import torch
from methods.tcdf_tools import findcauses, convert_weights_to_window_dag
import numpy as np

"""Adapted from https://github.com/M-Nauta/TCDF"""
def tcdf(data, cfg):
    """Loops through all variables in a dataset and return the discovered causes, time delays, losses, attention scores and variable names."""
    allcauses = dict()
    alldelays = dict()
    allreallosses=dict()
    allscores=dict()
    allweights = dict()

    columns = list(range(data.shape[1]))

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    epochs = cfg.epochs
    max_lag = cfg.max_lag
    optimizername = cfg.optimizername
    learningrate = cfg.learningrate
    dilation_coefficient = cfg.dilation_coefficient
    significance = cfg.significance
    loginterval= cfg.loginterval
    levels = cfg.levels
    seed = cfg.seed

    for c in columns:
        causes, causeswithdelay, realloss, scores, weights = findcauses(data, c, device=device, epochs=epochs, 
            kernel_size=max_lag+1, layers=levels, log_interval=loginterval, 
            lr=learningrate, optimizername=optimizername,
            seed=seed, dilation_c=dilation_coefficient, significance=significance)

        allweights[c]=weights
        allscores[c]=scores
        allcauses[c]=causes
        alldelays.update(causeswithdelay)
        allreallosses[c]=realloss

    return allcauses, alldelays, allreallosses, allscores, allweights


def tcdf_baseline(data_sample, cfg):
    # data preprocessing
    # expect data in dataframe where rows are timesteps
    # and columns are variables
    data = data_sample.to_numpy()
    allcauses, alldelays, allreallosses, allscores, allweights = tcdf(data, cfg)


    # if true, then we use the learned attention weights as scores
    #          Note that this is not the original behavior.
    # if false, then we build the binary result matrix according to 
    #           the original tcdf algorithm.
    if cfg.use_weights_directly:
        return convert_weights_to_window_dag(allweights)
    else:
        dag = np.zeros((data.shape[1], data.shape[1], cfg.max_lag+1))

        for effect, cause in alldelays:
            delay = alldelays[(effect, cause)]

            dag[effect, cause, delay] = 1

        return dag