import click
from pytorch_lightning import Trainer

from .train import MPII
import torch


@click.command()
@click.option("-d", "--data_path", help="dataset absolute path")
def main(data_path):
    torch.autograd.set_detect_anomaly(True)
    model = MPII(data_path=data_path)

    # most basic trainer, uses good defaults
    trainer = Trainer()
    trainer.fit(model)


if __name__ == "__main__":
    main()
