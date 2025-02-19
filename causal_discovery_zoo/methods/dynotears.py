from causalnex.structure.dynotears import from_numpy_dynamic, from_pandas_dynamic
import numpy  as np



def dynotears_baseline(data_sample,cfg):
    # do not accidentally break the formatting later due to overlapiing namings
    data_sample.columns = ["Var" + str(x) for x in range(len(data_sample.columns))]
    sm = from_pandas_dynamic(data_sample[:cfg.cut_at].reset_index(drop=True),p=cfg.max_lag)
    pred = reformat_dynotears(data_sample,sm,cfg.max_lag)
    return pred



def reformat_dynotears(data_sample,sm, max_lag):
    pred = list(sm.adjacency())
    # remove lag 0
    pred = [x for x in pred if x[0].split("_")[-1] != "lag0"]
    result = np.zeros((len(data_sample.columns),len(data_sample.columns),max_lag))
    mapping = {x:n for n, x in  enumerate(data_sample.columns)}
    # effect x cause x lag
    for x in pred: 
        # cause
        lag = x[0][-1]
        cause = x[0].split("_")[0]
        # effect: 
        for key in x[1].keys():
            assert key[-1] == "0", "Formatting ISSUE"
            effect = key.split("_")[0]
            value = x[1][key]["weight"]
            result[mapping[effect],mapping[cause], int(lag)-1] = value
    return result