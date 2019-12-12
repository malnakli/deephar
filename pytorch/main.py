import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import os
from .data import MpiiSinglePerson
from .losses import pose_regression_loss
from .models import MultitaskStemNet, PredictionBlockPS
from .config import mpii_sp_dataconf


class MPII(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # not the best model...
        self.stem = MultitaskStemNet()
        self.pred_blocks = PredictionBlockPS()

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
                dataset_path=os.getcwd(), dataconf=mpii_sp_dataconf, mode=1
            ),
            batch_size=32,
        )

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(
            MpiiSinglePerson(
                dataset_path=os.getcwd(), dataconf=mpii_sp_dataconf, mode=0
            ),
            batch_size=32,
        )


from pytorch_lightning import Trainer

model = MPII()

# most basic trainer, uses good defaults
trainer = Trainer()
trainer.fit(model)

