import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig

from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    RichModelSummary,
    RichProgressBar,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader

from additional_methods.causal_pretraining import Architecture_PL, lagged_batch_corr
import pickle
from torch.utils.data import Dataset
import lightning as L
import torch
from tools.tools import preprocessing_pipeline
import datetime
import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf


def preprocessing_to_finetune_ds(Y,Y_names,X, cfg):

    # On the fly data transformation. 
    label_stack = []
    for l in Y:
        lab = torch.Tensor(l)
        # works for 5 vars only. adapt if neccessary.
        assert lab.shape[1] == 5, "Check if 5 vars are used!"
        label_stack.append(lab)

    input_stack = []
    corr_stack = []
    for sample in Y_names: 
        sample_prep = X[[str(x) for x in sample]]
        check_trailing_nans = np.where(sample_prep.isnull().values.any(axis=1) == 0)[0]
        sample_prep = sample_prep[check_trailing_nans.min() : check_trailing_nans.max()+1]
        assert sample_prep.isnull().sum().max() == 0, "Check nans!"
        sample_prep = torch.Tensor(sample_prep.values).unsqueeze(0)
        corr = lagged_batch_corr(sample_prep, cfg.max_lag) # pretrained model takes 3 lags which we also use. 
        # Could be interesting to adapt this further.
        # This sometimes happens if the ts is constant.
        corr = torch.nan_to_num(corr,nan=0)
        corr_stack.append(corr[0])
        input_stack.append(sample_prep[0])

    return Finetune_DS(input_stack, label_stack, corr_stack, cfg.window_size)


# prep data.
class Finetune_DS(Dataset):

    # This randomly subselects a properly sized window of the time series. 


    def __init__(
        self, X, Y, corr, window_size
    ):
        self.X = X
        self.corr = corr
        self.Y = Y
        # This is the size this model was pretrained on before so we remain with this.
        self.window_size=window_size 
        self.random_draw = True

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        # quick and dirty data augmentation.
        if self.random_draw:
            random_selection = torch.randint(len(self.X[idx]) - (self.window_size + 1), (1,), dtype=torch.int64)[0]
            return (self.X[idx][random_selection:random_selection+self.window_size], self.corr[idx]), self.Y[idx]
        else:
            return None

    


class GeneratorDataModule(pl.LightningDataModule):
    def __init__(
        self,train_set, val_set,test_set,batch_size, 
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set


    def setup(self, stage):
        self.eeg_train = self.train_set
        self.eeg_val =  self.val_set
        self.eeg_test = self.test_set
        self.eeg_predict = self.test_set

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, num_workers=8, shuffle=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.eeg_val, batch_size=self.batch_size, num_workers=8, shuffle=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.eeg_test, batch_size=self.batch_size, num_workers=8, shuffle=True
        )

    def predict_dataloader(self):
        return DataLoader(
            self.eeg_predict, batch_size=self.batch_size, num_workers=8, shuffle=True
        )

    def teardown(self, stage):
        pass


@hydra.main(version_base=None, config_path="config", config_name="cp_finetuning.yaml")
def main(cfg: DictConfig):


    logger: TensorBoardLogger = hydra.utils.instantiate(cfg.tensorboard)
    logger.log_hyperparams(cfg)


    start = datetime.datetime.now()
    print("Loading data...")
    test_data = pickle.load(open(cfg.sample_path + cfg.ds_name + "/test.p", "rb"))
    train_data = pickle.load(open(cfg.sample_path + cfg.ds_name + "/train.p", "rb"))




    print("Prepare data...")
    Y,Y_names,X = preprocessing_pipeline(test_data,cfg)
    test_ds = preprocessing_to_finetune_ds(Y,Y_names,X, cfg)
    del Y,Y_names,X # remove as not needed.

    Y,Y_names,X = preprocessing_pipeline(train_data,cfg, train=True)
    train_ds = preprocessing_to_finetune_ds(Y,Y_names,X, cfg)
    del Y,Y_names,X # remove as not needed.


    data_set = GeneratorDataModule(train_set=train_ds,val_set=test_ds ,test_set=test_ds, batch_size=cfg.batch_size)


    print("Setup Mode...")
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)

    architecture = cfg.pretrained_path + cfg.cp_architecture + ".ckpt"
    model = Architecture_PL.load_from_checkpoint(architecture)
    # new last alyer and formatting.
    model.adapt_structure_to_rivers()


    model.optimizer_lr = cfg.optimizer_lr
    model.weight_decay = cfg.weight_decay


    inner_p = cfg.save_finetune_path + "/" +  str(start)
    checkpoint_callback = ModelCheckpoint(
        dirpath=inner_p,
        save_top_k=1,
        mode="max",
        monitor="f1_max_val",
        save_last=True,
    )

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, RichModelSummary(), RichProgressBar()],
    )

    trainer.fit(model=model, datamodule= data_set)
    print(datetime.datetime.now() - start)


    with open(inner_p + "/config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

if __name__ == "__main__":
    main()
# We use the best performing checkpoints from all std runs here. They can be downloaded here:
