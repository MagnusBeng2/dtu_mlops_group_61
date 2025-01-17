import torchvision
import warnings
import argparse
from typing import Optional
from pathlib import Path
import time
import pytorch_lightning as pl
import torch
import wandb
from datasets import load_from_disk
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from src.models.model import Model  # Import your custom LightningModule

# Warnings disabled
warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", category=UserWarning)
torchvision.disable_beta_transforms_warning()


def train(config: str, wandbkey: Optional[str] = None, debug_mode: bool = False):
    # Initialize W&B
    if wandbkey:
        wandb.login(key=wandbkey)
        wandb_mode = "online"
    else:
        wandb_mode = "disabled"

    wandb.init(
        project="mlops_exam_project",
        entity="chrillebon",
        config=config,
        mode=wandb_mode,
    )

    # Extract hyperparameters
    lr = wandb.config.get("lr", 0.001)
    epochs = wandb.config.get("epochs", 5)
    batch_size = wandb.config.get("batch_size", 16)
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
    print("Sample batch:", next(iter(DataLoader(trainset, batch_size=1, num_workers=2))))

    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=2, shuffle=True)
    testloader = DataLoader(valset, batch_size=batch_size, num_workers=2)

    print(f"Number of training batches: {len(trainloader)}")
    print(f"Number of validation batches: {len(testloader)}")
    print("Sample batch:", next(iter(trainloader)))

    # Initialize the model
    model = Model(lr=lr, batch_size=batch_size)

    # Configure W&B logger
    logger = pl.loggers.WandbLogger(project="mlops_exam_project", entity="chrillebon") if wandbkey else None
    if wandbkey:
        wandb.watch(model, log_freq=100)

    # Setup checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"models/checkpoints/{wandb.run.name if wandbkey else 'default_run'}",
        save_top_k=1,
        monitor="val_loss",
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
        max_epochs=1,  # Train for only one epoch
        limit_train_batches=1,  # Use only one batch for training
        limit_val_batches=1,  # Use only one batch for validation
        accelerator="cpu",  # Use CPU or GPU if available
        devices=1,  # Number of devices to use
        logger=logger,
        precision=32,  # Use full precision for simplicity
        num_sanity_val_steps=0,  # Skip validation sanity checks
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
