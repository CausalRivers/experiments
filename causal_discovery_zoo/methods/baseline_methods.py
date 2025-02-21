import numpy as np


def calc_lagged_cross_corr(df, max_lag=500, grain=1):
    """
    return Crosscorrelation items for each variables with each variable for each lag. 

    Return format: Stable, moved, lag
    Importantly: If the peak is above the max_lag, 
    the cause likely goes from  stable to moved and the other way arround.
    """
    product = []
    for column in df.columns:
        stack = []
        for x in range(-max_lag,max_lag+1, grain):
            stack.append(df.shift(-x).corrwith(df[column], axis=0).values.T)
        product.append(stack)
    return np.array(product).swapaxes(1,2)


def remove_edges_via_mean_values(d, cfg, return_means=False):
    """
    As rivers cannot flow from big to small (in most cases), 
    a baseline strategy is to simply remove these impossible 
    links (along with the diagonal) and keep the rest as prediction 
    As an additional restriction, we can only keep one link per node. 
    Here the strategy would be either to choose the biggest or the closest one
    input: pd.Dataframe, reverse_physical: Small to big possible
    outtput: Effect, Cause (nxn)
    """
    m = np.expand_dims(d.mean().values,axis=1).repeat(len(d.columns),axis=1)
    out = m > m.T if cfg.reverse_physical else m < m.T

    #for each column we choose the item which is closest to itself or the biggest available.
    if (cfg.filter_mode == "next")  and (cfg.name != "combo"):
        print("using next filter")
        m[~out] = m.max()+1
        out = out * (m == np.expand_dims(m.min(axis=0),axis=0).repeat(out.shape[0],axis=0))
    elif (cfg.filter_mode  == "biggest") and (cfg.name != "combo"):
        print("using biggest filter")
        m[~out] = 0
        out = out * (m == np.expand_dims(m.max(axis=0),axis=0).repeat(out.shape[0],axis=0))


    return  (out, m) if return_means else out


def cross_correlation_for_causal_discovery(d, cfg, return_corr=False):
    """ 
    Based on a cross correlation map, this decide which direction an error goes.
    """
    # we actually just have to calculate half here but keep it as it is cleaner.
    corr_map = calc_lagged_cross_corr(d,cfg.max_lag,1)
    out =  corr_map.argmax(axis=2).T > int(corr_map.shape[2] / 2)
    # restrict to the river with the higest cross correlation
    if (cfg.filter_mode == "corr") and (cfg.name != "combo"):
        print("using corr filter")
        peak = corr_map.max(axis=2)
        peak[~out] = 0
        out = out * (peak == np.expand_dims(
            peak.max(axis=0),axis=0).repeat(peak.shape[0],axis=0))


    return (out, corr_map) if return_corr else out


def combo_baseline(d, cfg):
    """ 
    This combines both baseline rules.
    """

    cc_out, corr_map = cross_correlation_for_causal_discovery(d, cfg, return_corr=True)
    rphy_out, m = remove_edges_via_mean_values(d, cfg, return_means=True)
    out = cc_out * rphy_out

    if (cfg.filter_mode == "next"):
        print("using next filter")
        m[~out] = m.max()+1
        out = out * (m == np.expand_dims(m.min(axis=0),axis=0).repeat(out.shape[0],axis=0))
    elif cfg.filter_mode == "biggest":
        print("using biggest filter")
        m[~out] = 0
        out = out * (m == np.expand_dims(m.max(axis=0),axis=0).repeat(out.shape[0],axis=0))
    elif cfg.filter_mode == "corr":
        print("using biggest corr")
        peak = corr_map.max(axis=2)
        peak[~out] = 0
        out = out * (peak == np.expand_dims(
            peak.max(axis=0),axis=0).repeat(peak.shape[0],axis=0))
    return out