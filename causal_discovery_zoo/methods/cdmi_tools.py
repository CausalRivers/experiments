import numpy as np
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.api import VAR
import pandas as pd
import pickle
from os import listdir
import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score
from methods.knockoff_tools import generate_knockoff


def remove_diagonal(T):
    # Takes in 3 dim tensor and removes diagonal of 2/3 dim.
    out = []
    for x in T:
        out.append(x[~np.eye(x.shape[0],dtype=bool)].reshape(x.shape[0],-1))
    return np.stack(out)


def eval_with_custom_data(eval_func,predictor,data,test_window, cfg):
    pred = eval_func(
            predictor,
            data= data,
            cfg= cfg,
        )
    errors = [calc_error(test_window.values[:,x],
                                    pred[:,x], cfg) for x in range(test_window.shape[1])]
    return pred, errors

def cdmi(original_data, predictor,eval_func, training_length, num_windows, cfg):
    """
    CDMI Wrapper to estimate the causal relationship 
    via a trained model and data intervention
    """
    start = 0
    residual_stack = [] 
    residual_intervention_stack = []
    window_preds = []
    data_samples = []
    knockoffs = []
    # multiple prediction windows to construct the residual distribution.
    for iteration in range(num_windows): 
        #print("Window:", iteration)
        # Generate an intervention for the window that includes ONLY the training interval and the testing interval.
        end = start + training_length + cfg.prediction_length
        test_window =  original_data.iloc[end-cfg.prediction_length:end].copy()
        int_data = get_intervention(original_data.iloc[start:end].copy(),cfg)
        knockoffs.append(int_data)
        data_samples.append([int_data,original_data.iloc[start:end]])
        # We here perform an intervention on the variable i and observe residual changes.
        # Accuracy without the intervention.
        no_intervention =  original_data.iloc[start:end].copy()
        pred, errors = eval_with_custom_data(
                eval_func,predictor,no_intervention,test_window, cfg)
        # Now we perform an intervention on the variable i and observe residual changes.
        intervention_error_stack = []
        intervention_prediction_stack = []
        for  i in original_data.columns:
            # ONLY replace the current ts with the intervention.
            single_intervention = original_data.iloc[start:end].copy()
            single_intervention[i] = int_data[i] 
            int_pred,int_errors = eval_with_custom_data(
                eval_func,predictor,single_intervention,test_window, cfg)
            intervention_error_stack.append(int_errors)
            intervention_prediction_stack.append(int_pred)

        # save changes
        window_preds.append([pred, intervention_prediction_stack, test_window.values])
        residual_intervention_stack.append(intervention_error_stack)
        residual_stack.append(errors)
        # step forward for new window
        start += cfg.step_size

    if cfg.save_intermediate: 
        pickle.dump(knockoffs, open(cfg.save_path +  "/knockoffs.p", "wb"))
        pickle.dump(residual_stack, open(cfg.save_path + "/default_resids.p", "wb"))
        pickle.dump(residual_intervention_stack, open(cfg.save_path + "/intervention_resids.p", "wb"))
        pickle.dump(window_preds, open(cfg.save_path + "/pred.p", "wb"))
        pickle.dump(data_samples, open(cfg.save_path + "/intervention.p", "wb"))


    return construct_statistics(np.array(residual_stack), np.array(residual_intervention_stack), cfg)



def construct_statistics(residual_stack, residual_intervention_stack, cfg):
    # run the specified significance test for all combos.
    decision_matrix = np.zeros(residual_intervention_stack.shape[1:])

    for intervention in range(residual_intervention_stack.shape[1]):
        for effect in range(residual_intervention_stack.shape[1]):
            stat = test_significance(residual_stack[:,intervention],
                                                                residual_intervention_stack[:,intervention,effect],cfg)
            decision_matrix[effect,intervention] = stat
    if cfg.normalize_effect_strength:
        print("Normalizing effect strengths")
        decision_matrix = (decision_matrix - decision_matrix.min()) / (decision_matrix.max() - decision_matrix.min())
    return decision_matrix

def calc_error(y_true,y_pred, cfg):
    print(y_true, y_pred)
    if cfg.error_metric == "mape":
        error  = mean_absolute_percentage_error(y_true,y_pred)
    elif cfg.error_metric == "mse":
        error  = mean_squared_error(y_true, y_pred)
    elif cfg.error_metric == "mae":
        error  =mean_absolute_error(y_true, y_pred)
    return error


def get_intervention(data,cfg):
    """
    int_data: pd.DataFrame
    i: int (knockoff variable)
    cfg: Hydra config
    """
    int_data = data.copy()
    # Generate sample specific intervention.
    if cfg.intervention_type == "knockoff":
        intervene = np.array(generate_knockoff(int_data.values))
        return pd.DataFrame(intervene, columns= int_data.columns, index= data.index)
    else: 
        for col in int_data.columns:
            if cfg.intervention_type == "mean":
                int_data[col] = np.random.normal(int_data[col].mean(), int_data[col].std(),len(int_data))
            elif cfg.intervention_type == "normal":
                int_data[col] =  np.random.normal(0, cfg.mean_std,len(int_data))
            elif cfg.intervention_type == "uniform":
                int_data[col] = np.random.uniform(int_data[col].min(), int_data[col].max(), len(int_data))
            elif cfg.intervention_type == "extreme":
                int_data[col] =  np.random.normal(100, cfg.mean_std,len(int_data))
            else:
                raise ValueError("Unknown intervention type")
        return int_data

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def abs_residual_increase(y_true, y_pred):
    # The lower the value the higher the "significance" (as for p values)
    return np.abs(y_true).sum() - np.abs(y_pred).sum() 


def test_significance(normal, inter, cfg):
    # The lower the higher the chance for a link
    if cfg.significance_test == "kolmo":
     _, stat = ks_2samp(normal, inter)
    elif cfg.significance_test == "kl_div":
        stat = kl_divergence(normal,inter)
    elif cfg.significance_test == "abs_error":
        stat = abs_residual_increase(normal, inter)

    else: 
        print("STAT TEST UNKNOWN")
        stat = None
    return stat