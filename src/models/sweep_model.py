import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import yaml
import argparse

import pytorch_lightning as pl
import torch
import wandb
from datasets import Dataset
from torch.utils.data import DataLoader

from src.models.model import Model


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        lr = config.lr
        epochs = config.epochs
        batch_size = config.batch_size

        logger = pl.loggers.WandbLogger(
            project="dtu_mlops_group_61", entity="mabbi-danmarks-tekniske-universitet-dtu"
        )

        model = Model(lr=lr, batch_size=batch_size)

        trainset = Dataset.load_from_disk("data/processed/train")
        testset = Dataset.load_from_disk("data/processed/validation")
        trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=8)
        testloader = DataLoader(testset, batch_size=batch_size, num_workers=8)

        trainer = pl.Trainer(
            default_root_dir="lightning_logs",
            limit_train_batches=0.1,
            limit_val_batches=0.1,
            max_epochs=epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=logger,
            precision=32,
            num_sanity_val_steps=0,
        )

        # Add this line to start training
        trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=testloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandbkey",
        default="6c90481a16aa4dc6b2905c30a6de1a230d6f427e",
        type=str,
        help="W&B API key",
    )
    parser.add_argument(
        "--config_path",
        default="./src/models/config/sweep_config.yaml",  # Adjusted path
        type=str,
        help="Path to the sweep configuration file",
    )
    args = parser.parse_args()
    wandbkey = args.wandbkey
    config_path = args.config_path

    # Load sweep configuration from the YAML file
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            sweep_config = yaml.safe_load(file)
    else:
        raise FileNotFoundError(f"Sweep configuration file not found at: {config_path}")

    wandb.login(key=wandbkey)  # Input API key for W&B for Docker
    project = "dtu_mlops_group_61"
    entity = "mabbi-danmarks-tekniske-universitet-dtu"
    anonymous = None
    mode = "online"

    # Use the loaded sweep_config
    sweep_id = wandb.sweep(
        sweep=sweep_config, project="dtu_mlops_group_61", entity="mabbi-danmarks-tekniske-universitet-dtu"
    )
    wandb.agent(sweep_id, train, count=5)
