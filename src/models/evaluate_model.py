import os
import pytorch_lightning as pl
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from src.models.model import Model

# Warnings disabled
import torchvision
import warnings
warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", message="The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta")
warnings.filterwarnings("ignore", message="Passing a tuple of `past_key_values` is deprecated")
torchvision.disable_beta_transforms_warning()

def get_latest_checkpoint(base_dir="lightning_logs"):
    # Find all version folders
    versions = sorted(
        (v for v in os.listdir(base_dir) if v.startswith("version_")),
        key=lambda x: int(x.split("_")[-1]),
        reverse=True,
    )

    # Search for checkpoints in the most recent versions
    for version in versions:
        checkpoints_dir = os.path.join(base_dir, version, "checkpoints")
        if os.path.exists(checkpoints_dir) and os.listdir(checkpoints_dir):
            # Return the first checkpoint found
            return os.path.join(checkpoints_dir, os.listdir(checkpoints_dir)[0])

    raise FileNotFoundError("No checkpoints found in any version folders.")


if __name__ == "__main__":
    # Get the latest checkpoint path
    checkpoint_path = get_latest_checkpoint()

    # Load the model from the latest checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = Model.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # Load the test dataset
    testset = Dataset.load_from_disk("data/processed/validation")
    testloader = DataLoader(testset.select(range(50)), batch_size=1, num_workers=12)  # Use first 500 examples

    # Configure the trainer
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)

    # Evaluate the model
    results = trainer.test(model=model, dataloaders=testloader, verbose=True)
    print("Evaluation results:", results)
