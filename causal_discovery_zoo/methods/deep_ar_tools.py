import numpy as np
from gluonts.evaluation.backtest import make_evaluation_predictions
from sklearn.metrics import mean_squared_error, mean_absolute_error
from gluonts.dataset.common import ListDataset
from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
import pickle
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput
import os


def train_deep_ar_estimator(df,cfg):
    """
    df : pandas.DataFrame
    training_length : int
    cfg: Hydra config.
    
    """
    train_ds = ListDataset(
        [
            {
            'target': df.values.T.tolist(),
            "start": df.index[0]
            }
        ],
        freq=cfg.freq,
        one_dim_target=False
    )

    

    #Training model if necessary
    if not cfg.use_cached_model:
        # create estimator
        estimator = DeepAREstimator(
            prediction_length=cfg.prediction_length,
            context_length=cfg.context_length,
            freq=cfg.freq,
            num_layers=cfg.num_layers,
            num_cells=cfg.num_cells,
            dropout_rate=cfg.dropout_rate,
            trainer=Trainer(
                ctx=cfg.device,
                epochs=cfg.epochs,
                hybridize=False,
                learning_rate=cfg.learning_rate,
                batch_size=cfg.batch_size
            ),
            distr_output=MultivariateGaussianOutput(dim=df.shape[1])
        )

        print("Training forecasting model....")
        M = estimator.train(train_ds)
        if cfg.save_intermediate:
            pickle.dump(M, open(cfg.save_path + "/model.p", "wb"))
        return M
    
    else: # load alternatively
        print("Loading forecasting model....")
        M = pickle.load(open(cfg.save_path + "/model.p", "rb"))
        return M

def eval_deep_ar_estimator(predictor,data,cfg):
    """
    predictor: model
    data: data that is required to make a prediction for the next window.
    cfg: Hydra config file
    return prediction for the next forecast window.
    """
    test_ds = ListDataset(
                            [{
                                "target": data.values.T.tolist(),
                            "start": data.index[0]
                            }],
                            freq=cfg.freq,
                            one_dim_target=False,
                        )
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=cfg.num_samples,  # number of sample paths we want for evaluation
    )
    forecasts = list(forecast_it)
    y_pred = []
    for i in range(cfg.num_samples):
        y_pred.append(forecasts[0].samples[i].transpose().tolist())
    y_pred = np.array(y_pred)
    return y_pred.mean(axis=0).T
   