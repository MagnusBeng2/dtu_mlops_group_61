import torchvision
import warnings
import argparse
from typing import Optional
from pathlib import Path
import time
import pytorch_lightning as pl
import torch
import os
import wandb
from datasets import load_from_disk
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from src.models.model import Model

# Warnings disabled
warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", message="The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta")
warnings.filterwarnings("ignore", message="Passing a tuple of `past_key_values` is deprecated")
torchvision.disable_beta_transforms_warning()

# Get correct version directory
def get_next_version(base_dir="lightning_logs"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return os.path.join(base_dir, "version_0")

    versions = [
        int(v.split("_")[-1]) for v in os.listdir(base_dir) if v.startswith("version_")
    ]
    next_version = max(versions, default=-1) + 1
    return os.path.join(base_dir, f"version_{next_version}")

def train(config: str, wandbkey: Optional[str] = None, debug_mode: bool = False):
    # Initialize W&B
    wandb.login(key=wandbkey)
    wandb.init(
        project="dtu_mlops_group_61",
        entity="mabbi-danmarks-tekniske-universitet-dtu",
        config=config,
    )

    # Extract hyperparameters
    lr = wandb.config.get("lr", 0.001)
    epochs = wandb.config.get("epochs", 5)
    batch_size = wandb.config.get("batch_size", 1)
    seed = wandb.config.get("seed", None)

    if seed is not None:
        torch.manual_seed(seed)

    # Load tokenized datasets
    root_dir = Path(__file__).resolve().parents[2]
    train_path = root_dir / "data/processed/train"
    val_path = root_dir / "data/processed/validation"

    start_time = time.time()
    trainset = load_from_disk(str(train_path))
    valset = load_from_disk(str(val_path))
    print(f"Loaded tokenized datasets in {time.time() - start_time:.2f} seconds.")

    print(f"Training dataset size: {len(trainset)}")
    print(f"Validation dataset size: {len(valset)}")

    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=12, shuffle=True)
    testloader = DataLoader(valset, batch_size=batch_size, num_workers=0)

    # Initialize the model
    model = Model(lr=lr, batch_size=batch_size)

    # Configure W&B logger
    logger = pl.loggers.WandbLogger(
        project="dtu_mlops_group_61",
        entity="mabbi-danmarks-tekniske-universitet-dtu"
    )
    wandb.watch(model, log_freq=100)

    # Setup checkpointing
    checkpoint_dir = os.path.join(get_next_version(), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch={epoch}-step={step}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    # Determine debug mode limits dynamically
    if debug_mode:
        limit_train_batches = max(1 / len(trainloader), 0.1)
        limit_val_batches = max(1 / len(testloader), 0.1)
    else:
        limit_train_batches = 1.0
        limit_val_batches = 1.0

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        limit_train_batches=1,
        limit_val_batches=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        precision=32,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,  # Disable sanity check
    )

    # Log epoch losses and accuracy to W&B
    trainer.callbacks.append(
        pl.callbacks.LambdaCallback(
            on_train_epoch_end=lambda trainer, pl_module: wandb.log({
                "train_loss": trainer.callback_metrics.get("train_loss", 0),
                "train_acc": trainer.callback_metrics.get("train_acc", 0),
            }),
            on_validation_epoch_end=lambda trainer, pl_module: wandb.log({
                "val_loss": trainer.callback_metrics.get("val_loss", 0),
                "val_acc": trainer.callback_metrics.get("val_acc", 0),
            })
        )
    )

    # Train the model
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=testloader)
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
