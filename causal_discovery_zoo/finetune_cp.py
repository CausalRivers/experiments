import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig

from lightning.pytorch.callbacks import (
    RichModelSummary,
    RichProgressBar,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader

from methods.causal_pretraining import Architecture_PL, lagged_batch_corr
from torch.utils.data import Dataset
import lightning as L
import torch
from tools.tools import load_joint_samples, standard_preprocessing
import datetime
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
import pandas as pd




# prep data.
class Finetune_DS(Dataset):

    # This randomly subselects a properly sized window of the time series.

    def __init__(self, X, Y, corr, window_size):
        self.X = X
        self.corr = corr
        self.Y = Y
        # This is the size this model was pretrained on before so we remain with this.
        self.window_size = window_size
        self.random_draw = True

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        # quick and dirty data augmentation.
        if self.random_draw:
            random_selection = torch.randint(
                len(self.X[idx]) - (self.window_size + 1), (1,), dtype=torch.int64
            )[0]
            return (
                self.X[idx][random_selection : random_selection + self.window_size],
                self.corr[idx],
            ), self.Y[idx]
        else:
            return None


def preprocessing_to_finetune_ds(X,Y, cfg):
    # On the fly data transformation.
    
    
    assert Y[0].shape[0] == 5, "Only implemented for same nvars as pretraining"
    label_stack = [torch.Tensor(y.values) for y in Y]


    input_stack = []
    corr_stack = []
    for sample in X:
        
        assert sample.isnull().sum().max() == 0, "Check nans!"
        sample_prep = torch.Tensor(sample.values).unsqueeze(0)
        corr = lagged_batch_corr(
            sample_prep, 3
        )  # pretrained model takes 3 lags which we also use.
        # Could be interesting to adapt this further.
        
        # Sometimes cross corr bricks if a ts is missshaped (constant for a long time)
        corr = torch.nan_to_num(corr, nan=0)
        corr_stack.append(corr[0])
        input_stack.append(sample_prep[0])

    return Finetune_DS(input_stack, label_stack, corr_stack, cfg.window_size)
    



class GeneratorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_set,
        val_set,
        test_set,
        batch_size,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

    def setup(self, stage):
        self.eeg_train = self.train_set
        self.eeg_val = self.val_set
        self.eeg_test = self.test_set
        self.eeg_predict = self.test_set

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, num_workers=8, shuffle=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.eeg_val, batch_size=self.batch_size, num_workers=8, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.eeg_test, batch_size=self.batch_size, num_workers=8, shuffle=False
        )

    def predict_dataloader(self):
        return DataLoader(
            self.eeg_predict, batch_size=self.batch_size, num_workers=8, shuffle=False
        )

    def teardown(self, stage):
        pass


@hydra.main(version_base=None, config_path="config", config_name="cp_finetuning.yaml")
def main(cfg: DictConfig):

    logger: TensorBoardLogger = hydra.utils.instantiate(cfg.tensorboard)
    logger.log_hyperparams(cfg)
    
    
    
    start = datetime.datetime.now()
    print("Loading data...")


    test_data, test_labels = load_joint_samples(
        cfg, preprocessing=standard_preprocessing
    )
    print(len(test_data))
    test_ds = preprocessing_to_finetune_ds(test_data[:],test_labels, cfg)
    del test_data, test_labels
    
    
    cfg.label_path =  "../../datasets/random_5/bav.p"
    cfg.data_path = "../../product/rivers_ts_bavaria.csv"

    train_data, train_labels = load_joint_samples(
        cfg, preprocessing=standard_preprocessing
    )

    train_val_split = int(len(train_data) * 0.9)
    print(train_val_split)
        
    train_ds = preprocessing_to_finetune_ds(train_data[:train_val_split],train_labels[:train_val_split], cfg)
    val_ds = preprocessing_to_finetune_ds(train_data[train_val_split:],train_labels[train_val_split:], cfg)
    del train_data, train_labels
    


    data_set = GeneratorDataModule(
        train_set=train_ds, val_set=val_ds, test_set=test_ds, batch_size=cfg.batch_size
    )

    print("Setup Mode...")
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)
    architecture = cfg.pretrained_path + cfg.cp_architecture + ".ckpt"
    print(architecture)
    model = Architecture_PL.load_from_checkpoint(architecture)
    # new last alyer and formatting.
    print(model.d_ff)
    model.adapt_structure_to_rivers()

    model.optimizer_lr = cfg.optimizer_lr
    model.weight_decay = cfg.weight_decay

    inner_p = cfg.save_finetune_path + "/" + str(start)
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

    trainer.fit(model=model, datamodule=data_set)
    print(datetime.datetime.now() - start)

    test = trainer.test(datamodule=data_set, ckpt_path="best")
    val = trainer.validate(datamodule=data_set, ckpt_path="best")
    test = pd.DataFrame(test).to_csv(inner_p + "/test_performance.csv")
    val = pd.DataFrame(val).to_csv(inner_p + "/val_performance.csv")

    with open(inner_p + "/config.yaml", "w") as f:
        OmegaConf.save(cfg, f)


if __name__ == "__main__":
    main()
