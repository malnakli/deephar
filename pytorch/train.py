import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import os
from .data import MpiiSinglePerson
from .losses import pose_regression_loss
from .models import MultitaskStemNet, PredictionBlockPS
from .config import mpii_sp_dataconf
from torchvision import transforms


class MPII(pl.LightningModule):
    def __init__(self, data_path=os.getcwd()):
        super().__init__()
        # not the best model...
        self.stem = MultitaskStemNet()
        self.pred_blocks = PredictionBlockPS()
        self.data_path = data_path

    def forward(self, x):
        x = self.stem(x)
        x = self.pred_blocks(x)
        return x

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss_f = pose_regression_loss("l1l2bincross", 0.01)
        loss = loss_f(y, y_hat)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.RMSprop(self.parameters(), lr=0.02)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(
            MpiiSinglePerson(
                dataset_path=self.data_path,
                dataconf=mpii_sp_dataconf,
                y_dictkeys=["pose"],
                num_predictions=8,
                mode=1,
                transform=transforms.ToTensor(),
            ),
            batch_size=32,
            shuffle=False,
        )

    # @pl.data_loader
    # def val_dataloader(self):
    #     # OPTIONAL
    #     return DataLoader(
    #         MpiiSinglePerson(
    #             dataset_path=self.data_path,
    #             dataconf=mpii_sp_dataconf,
    #             y_dictkeys=["pose", "afmat", "headsize"],
    #             mode=2,
    #         ),
    #         batch_size=32,
    #         shuffle=False,
    #     )

