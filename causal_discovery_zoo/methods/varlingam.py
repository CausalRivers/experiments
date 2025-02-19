import numpy as np
from statsmodels.tsa.api import VAR
import lingam

def varlingam_baseline(data_sample, cfg, cut_at=10000):

    model = lingam.VARLiNGAM(lags=cfg.max_lag, criterion=None)
    # bricks for very long time series so we have to shorten.
    try:
        model.fit(data_sample[:cut_at])
        # remove instant as its not available
        pred = np.abs(model.adjacency_matrices_[1:])
        pred = np.transpose(pred, axes=[1,2,0])
        # put lag at the back for consistency
    except:
        print("Fit failed")
        pred = np.zeros((data_sample.shape[1], data_sample.shape[1], cfg.max_lag))

    if np.isnan(np.array(pred)).sum() > 0:
        print("ISSUE detected, prediction set to 0") 
        pred = np.zeros(pred.shape)

    return pred
