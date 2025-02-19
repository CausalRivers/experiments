import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR


def var_baseline(d, cfg):
    """ 
    Simple Granger based strategy that selects based on absolute parameter values.
    """
    


    d.index = pd.DatetimeIndex(d.index.values,
                                freq=d.index.inferred_freq)
        
    
    n_vars = d.values.shape[-1]
    try: #For some samples, data is bricked to have constant TS. If this happens we predict 0
    # fit var with appropriate max lags
        res = VAR(d).fit(cfg.max_lag)
    # convert to bool and throw away intersection
    # !In the context of rivers, negative correlation do not really make sense.
    # I guess trying both is fair
        pred = res.params[1:]

        if cfg.var_absolute_values:
            pred = np.abs(pred)
        # reformat to original caused causing lag:
        # :) einsum needed i guess
        pred = np.stack(
            [pred.values[:, x].reshape(cfg.max_lag, n_vars).T for x in range(pred.shape[1])]
        )
    except:
        pred = np.zeros((n_vars,n_vars,cfg.max_lag))

    return pred