import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from torchmetrics.classification import BinaryF1Score

from methods.cp_models.gru import gru
from methods.cp_models.informer import transformer

# Super stripped down version of the CP repo.

def causal_pretraining_baseline(X,cfg):


    # some loading schemes
    model = Architecture_PL.load_from_checkpoint(cfg.weight_path + cfg.architecture + ".ckpt")
    if cfg.use_river_finetune:
        #TODO there has to be a better way here.
        model = Architecture_PL.load_from_checkpoint(cfg.weight_path + "unidirectional" + ".ckpt")
        model.adapt_structure_to_rivers()
        checkpoint = torch.load(cfg.finetune_path, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])

    M = model.model
    M = M.eval()
    M = M.to("cuda")

    n_vars = X.shape[1]
    sample_prep = pad(torch.Tensor(X.values).unsqueeze(0).to("cuda"))
    #Has to be provided.
    corr = lagged_batch_corr(sample_prep, 3)


    # sometimes there is a constant ts which results in nan corr. replace this with 0
    corr = torch.nan_to_num(corr,nan=0)
    sample_prep, corr = reshape_long_ts(sample_prep, corr)


    pred =[]
    for x in range(0,corr.shape[0], cfg.batch_size):
        output = torch.sigmoid(M((sample_prep[x:x+cfg.batch_size], corr[x:x+cfg.batch_size]))).to("cpu").detach().numpy()
        # Sometimes this is an issue as with constant values this can produce overflow. catch this here.
        if np.isnan(output).sum() > 0:
            print("ISSUE, divergence of output? Possible but should be rare!") 
            output = torch.zeros(output.shape)
        pred.append(output)
    pred = np.concatenate(pred,axis=0)
    if cfg.use_river_finetune and (X.shape[1] != 5):
        pred = pred[:, :X.shape[1], :X.shape[1]]
    else:
        if n_vars != 5:
            pred = pred[:,:X.shape[1], :X.shape[1], :]


    if cfg.batch_aggregation == "mean":
        pred = np.array(pred).mean(axis=0)
    elif cfg.batch_aggregation == "max":
        pred = np.array(pred).max(axis=0)       
    return pred



#### Helpers ____________________________________


def reshape_long_ts(sample, corr):
    """
    # This speeds up the process and prevents overflow of the transformer
    """
    if (sample.shape[1] > 600): #and cfg.cp_architecture == "transformer":
        # we run multiple subsets of the data as batch and mean over the result. 
        sample = sample[:,: -int(sample.shape[1] % 600), :]
        sample = sample.reshape((int(sample.shape[1] / 600),600,sample.shape[2]))
        corr = corr.repeat(len(sample),1,1)
    return sample,corr


def pad(sample_prep):
    """
    Cp weights are for 5 variables.
    """
    if sample_prep.shape[2] != 5:
        sample_prep = torch.concat(
            [sample_prep[0, :, :], torch.normal(0, 0.1, (len(sample_prep[0]), 5-sample_prep.shape[2])).to("cuda")], axis=1
        )
        sample_prep = sample_prep.unsqueeze(0)
    else:
        sample_prep = sample_prep
    return sample_prep



class Architecture_PL(pl.LightningModule):
    def __init__(
        self,
        n_vars=3,
        max_lags=3,
        trans_max_ts_length=600,
        mlp_max_ts_length=600,
        model_type="unidirectional",
        corr_input=True,
        loss_type="ce",
        val_metric="ME",
        regression_head=False,
        link_thresholds=[0.25, 0.5, 0.75],
        corr_regularization=False,
        distinguish_mode=False,
        full_representation_mode=False,
        optimizer_lr=1e-4,
        weight_decay=0.01,
        d_model=32,
        n_heads=2,
        num_encoder_layers=2,
        d_ff=128,
        dropout=0.05,
        distil=True,
        # gru specifics
        gruU_hidden_size1=10,
        gruU_hidden_size2=10,
        gruU_hidden_size3=10,
        gruU_num_layers=10,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.n_vars = n_vars
        self.max_lags = max_lags

        self.loss_type = loss_type
        self.val_metric = val_metric
        self.optimizer_lr = optimizer_lr
        self.regression_head = regression_head
        self.corr_input = corr_input
        self.weight_decay = weight_decay
        self.link_thresholds = link_thresholds
        self.corr_regularization = corr_regularization
        self.trans_max_ts_length = trans_max_ts_length
        self.mlp_max_ts_length = mlp_max_ts_length
        self.full_representation_mode = full_representation_mode
        self.loss_scaling = {}
        self.distinguish_mode = distinguish_mode
        self.F1 =   [BinaryF1Score(th).to("cuda") for th in np.arange(0,1.1,0.1)]
        # specific for the model type not used all the time
        self.model_type = model_type
        self.d_ff = d_ff

        if self.model_type == "unidirectional":
            self.model = gru(
                max_lags=self.max_lags,
                n_vars=self.n_vars,
                hidden_size1=gruU_hidden_size1,
                hidden_size2=gruU_hidden_size2,
                hidden_size3=gruU_hidden_size3,
                num_layers=gruU_num_layers,
                corr_input=self.corr_input,
                regression_head=self.regression_head,
                direction=self.model_type,
            )

        elif self.model_type == "transformer":
            self.model = transformer(
                n_vars=n_vars,
                d_model=d_model,
                max_lags=max_lags,
                n_heads=n_heads,
                num_encoder_layers=num_encoder_layers,
                d_ff=d_ff,
                dropout=dropout,
                distil=distil,
                max_length=trans_max_ts_length,
                regression_head=self.regression_head,
                corr_input=corr_input,
            )
        else:
            print("MODEL TYPE NOT KNOWN!")

        self.classifier_loss = self.classifier_loss_init()


    def classifier_loss_init(self):
        if self.loss_type == "mse":
            print("init with MSE")
            return nn.MSELoss()
        if self.loss_type == "mae":
            print("init with mae")
            return nn.L1Loss()
        elif self.loss_type == "bce":
            return nn.BCEWithLogitsLoss()
        else:
            return None

    def calc_log_F1_metrics(self, y_class, lab_class, name="no_name"):
        out_d = {}

        curve = torch.Tensor([f1(y_class, lab_class) for f1 in self.F1])
        f1_max = torch.max(curve)
        out_d["f1_max"  "_" + name] = f1_max
        out_d["f1_max_th"  "_" + name] = torch.argmax(curve)
        self.log_dict(out_d, sync_dist=True, prog_bar=True)

    def training_step(self, batch, batch_idx):

        inputs, labels = batch
        # Change input representation for the MLP.
        y_ = self.model(inputs)
        proba = torch.sigmoid(y_)
        loss = self.classifier_loss(y_, labels)
        self.calc_log_F1_metrics(proba, labels, name="train")
        self.log("tr_output_mean", proba.mean(), sync_dist=True, prog_bar=True)

        self.log(
            "tr_class_loss",
            loss.type("torch.DoubleTensor"),
            sync_dist=True,
            prog_bar=True,
        )
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def non_train_step(self, batch, name="no_name"):
        inputs, labels = batch
        y_ = self.model(inputs)
        proba = torch.sigmoid(y_)
        loss = self.classifier_loss(y_, labels)

        self.log(
            name + "_class_loss",
            loss,
            sync_dist=True,
            prog_bar=True,
        )
        self.calc_log_F1_metrics(proba, labels, name=name)
        self.log(name + "_output_mean", proba.mean(), sync_dist=True, prog_bar=True)


    def validation_step(self, batch, _):
        self.non_train_step(batch, name="val")

    def test_step(self, batch, _):
        self.non_train_step(batch, name="test")

    def configure_optimizers(self):
        optim = opt.AdamW(
            self.model.parameters(),
            lr=self.optimizer_lr,  # betas= self.betas,
            weight_decay=self.weight_decay,
        )
        schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.2)
        return [optim], [{"scheduler": schedule, "monitor": "train_loss"}]


    def adapt_structure_to_rivers(self):
        if self.model_type == "transformer":
            self.model.fc2 = torch.nn.Linear(self.model.d_ff, 25)
        else:
            self.model.fc3 = torch.nn.Linear(self.model.hidden_size3, 25)
        def new_reformat(x):
                return torch.reshape(x, (x.shape[0], 5,5))
        self.model.reformat = new_reformat





def lagged_batch_corr(points, max_lags):
    # calculates the autocovariance matrix with a batch dimension
    # lagged variables are concated in the same dimension.
    # input (B, time, var)
    # roll to calculate lagged cov:
    # We dont use the batched component here so might be uneccesary complicated..


    B, N, D = points.size()

    # we roll the data and add it together to have the lagged versions in the table
    stack = torch.concat(
        [torch.roll(points, x, dims=1) for x in range(max_lags + 1)], dim=2
    )

    mean = stack.mean(dim=1).unsqueeze(1)
    std = stack.std(dim=1).unsqueeze(1)
    diffs = (stack - mean).reshape(B * N, D * (max_lags + 1))

    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(
        B, N, D * (max_lags + 1), D * (max_lags + 1)
    )

    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    # make correlation out of it by dividing by the product of the stds
    corr = bcov / (
        std.repeat(1, D * (max_lags + 1), 1).reshape(
            std.shape[0], D * (max_lags + 1), D * (max_lags + 1)
        )
        * std.permute((0, 2, 1))
    )
    # we can remove backwards in time links. (keep only the original values)
    return corr[:, :D, D:]  # (B, D, D)