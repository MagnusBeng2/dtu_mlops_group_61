import argparse
from typing import Optional
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
from datasets import load_from_disk
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from src.models.model import Model


def train(config: str, wandbkey: Optional[str] = None, debug_mode: bool = False):
    # Initialize W&B
    if wandbkey:
        wandb.login(key=wandbkey)  # Input API key for wandb for Docker
        wandb_mode = "online"
    else:
        wandb_mode = "disabled"

    wandb.init(
        project="mlops_exam_project",
        entity="chrillebon",
        config=config,
        mode=wandb_mode,
    )

    # Extract hyperparameters from W&B
    lr = wandb.config.get("lr", 0.001)
    epochs = wandb.config.get("epochs", 10)
    batch_size = wandb.config.get("batch_size", 32)
    seed = wandb.config.get("seed", None)

    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Load datasets
    root_dir = Path(__file__).resolve().parents[2]
    train_path = root_dir / "data/processed/train"
    val_path = root_dir / "data/processed/validation"

    trainset = load_from_disk(train_path)
    valset = load_from_disk(val_path)

    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=4, shuffle=True)
    testloader = DataLoader(valset, batch_size=batch_size, num_workers=4)

    # Initialize the model
    model = Model(lr=lr, batch_size=batch_size)

    # Configure W&B logger
    logger = (
        pl.loggers.WandbLogger(project="mlops_exam_project", entity="chrillebon")
        if wandbkey
        else None
    )
    if wandbkey:
        wandb.watch(model, log_freq=100)

    # Setup checkpointing
    checkpoint_callback = ModelCheckpoint(dirpath="models/checkpoints", save_top_k=1, monitor="val_loss")

    # Determine accelerator
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # Debug mode limits
    limit = 0.1 if debug_mode else 1.0

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        accelerator=accelerator,
        devices=1,  # Adjust devices based on your setup
        logger=logger,
        limit_train_batches=limit,
        limit_val_batches=limit,
        profiler="simple" if debug_mode else None,
    )

    # Train the model
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=testloader)

    # Save the model
    torch.save(model.state_dict(), "models/epoch=final.pt")
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="src/models/config/default_params.yaml",
        type=str,
        help="Configuration file with hyperparameters",
    )
    parser.add_argument("--wandbkey", default=None, type=str, help="W&B API key")
    parser.add_argument(
        "--debug_mode", action="store_true", help="Run only 10 percent of data"
    )

    args = parser.parse_args()
    train(args.config, args.wandbkey, args.debug_mode)
