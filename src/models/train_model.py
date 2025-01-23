import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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
from torch.utils.data import DataLoader, DistributedSampler
from src.models.model import Model
from torch.utils.data import RandomSampler

# Warnings disabled
warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", message="The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta")
warnings.filterwarnings("ignore", message="Passing a tuple of `past_key_values` is deprecated")
torchvision.disable_beta_transforms_warning()

# Set random seed
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_next_version(base_dir="lightning_logs"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return os.path.join(base_dir, "version_0")

    versions = [
        int(v.split("_")[-1]) for v in os.listdir(base_dir) if v.startswith("version_")
    ]
    next_version = max(versions, default=-1) + 1
    return os.path.join(base_dir, f"version_{next_version}")

def train(args):
    debug_mode = args.debug_mode

    # Initialize W&B
    if args.wandbkey:
        wandb.login(key=args.wandbkey)

    # Detect distributed training via torchrun
    is_distributed = "LOCAL_RANK" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if is_distributed:
        torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        torch.cuda.set_device(local_rank)

    # If wandb.config is active, use it for hyperparameters; otherwise, use args
    wandb.init(
        project="dtu_mlops_group_61",
        entity="mabbi-danmarks-tekniske-universitet-dtu",
        config={
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
    )
    config = wandb.config  # Use Wandb config for both manual runs and sweeps

    # Extract hyperparameters from Wandb config
    lr = config.lr
    epochs = config.epochs
    batch_size = config.batch_size
    seed = config.seed

    # Set the random seed
    set_seed(seed)

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

    # Define Distributed Sampler
    train_sampler = DistributedSampler(trainset) if is_distributed else RandomSampler(trainset)
    val_sampler = DistributedSampler(valset) if is_distributed else None

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        num_workers=12,
        sampler=train_sampler,
    )
    testloader = DataLoader(
        valset,
        batch_size=batch_size,
        num_workers=4,
        sampler=val_sampler,
    )

    # Initialize the model
    model = Model(lr=lr, batch_size=batch_size)

    # Configure W&B logger
    logger = pl.loggers.WandbLogger(
        project="dtu_mlops_group_61",
        entity="mabbi-danmarks-tekniske-universitet-dtu",
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

    print(f"ModelCheckpoint dirpath: {checkpoint_callback.dirpath}")
    print(f"Logger save_dir: {logger.save_dir if hasattr(logger, 'save_dir') else 'Logger save_dir not set'}")

    # Initialize Trainer
    trainer = pl.Trainer(
        default_root_dir="lightning_logs",
        max_epochs=epochs,
        limit_train_batches=0.1 if debug_mode else 1.0,
        limit_val_batches=0.1 if debug_mode else 1.0,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.gpus if args.gpus > 1 else 1,
        strategy="ddp" if is_distributed else "auto",
        logger=logger,
        precision=32,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
    )

    # Train the model
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=testloader)
    print("Training complete!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--wandbkey", default=None, type=str, help="W&B API key")
    parser.add_argument("--debug_mode", action="store_true", help="Run only a fraction of the dataset for quick testing")
    parser.add_argument("--gpus", default=0, type=int, help="Number of GPUs to use")

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
