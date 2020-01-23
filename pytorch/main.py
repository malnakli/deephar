import click
from pytorch_lightning import Trainer

from .train import MPII
import torch


@click.command()
@click.option("-d", "--data_path", help="dataset absolute path")
def main(data_path):
    torch.autograd.set_detect_anomaly(True)

    if torch.cuda.is_available():
        gpus = 1 
        device = torch.device("cuda:0")
    else:
        gpus = None
        device = 'cpu'

    model = MPII(data_path=data_path,device=device)

    # most basic trainer, uses good defaults
    trainer = Trainer(gpus=gpus)
    trainer.fit(model)


if __name__ == "__main__":
    main()
