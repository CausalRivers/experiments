from tigramite.independence_tests.parcorr_wls import ParCorrWLS, ParCorr
import numpy as np
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
#from tigramite.independence_tests.gpdc import GPDC



def pcmci_baseline(data_sample,cfg):

    if cfg.ci_test == "nonlinear": 
        print("NOT IMPLEMENTED")
    else:
        c_test  = ParCorr()


    dataframe = pp.DataFrame(data_sample.values[:cfg.cut_at],
                            datatime = np.arange(len(data_sample)), 
                            var_names=data_sample.columns)
    pcmci = PCMCI(
        dataframe=dataframe, 
        cond_ind_test=c_test,
        verbosity=1)

    pcmci.verbosity = 0
    results = pcmci.run_pcmci(tau_max=cfg.max_lag, tau_min=1, pc_alpha=None)
    pred = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')

    pred = results["p_matrix"]
    # remove instant link
    pred = pred[:,:,1:]
    pred = np.swapaxes(pred,0,1)
    # reverse as these are p values.
    pred = 1 - pred
    if np.isnan(pred).sum() > 0:
        print("ISSUE detected, prediction set to 0") 
        pred = np.zeros(pred.shape)
    return pred


