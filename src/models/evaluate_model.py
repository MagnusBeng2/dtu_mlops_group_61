import os
import pytorch_lightning as pl
import torch
from datasets import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from src.models.model import Model

def get_latest_checkpoint(logs_dir: str = "lightning_logs") -> str:
    """
    Get the path to the latest checkpoint in the lightning_logs directory.

    Parameters
    ----------
    logs_dir : str
        The root directory where the logs are saved.

    Returns
    -------
    str
        The path to the latest checkpoint.
    """
    if not os.path.exists(logs_dir):
        raise FileNotFoundError(f"The directory {logs_dir} does not exist.")

    # Find the latest version folder (e.g., version_0, version_1, ...)
    versions = [
        os.path.join(logs_dir, d)
        for d in os.listdir(logs_dir)
        if d.startswith("version_") and os.path.isdir(os.path.join(logs_dir, d))
    ]
    if not versions:
        raise ValueError(f"No version directories found in {logs_dir}.")
    
    latest_version = max(versions, key=lambda v: int(v.split("_")[-1]))

    # Find the latest checkpoint in the version folder
    checkpoints_dir = os.path.join(latest_version, "checkpoints")
    if not os.path.exists(checkpoints_dir):
        raise FileNotFoundError(f"No checkpoints directory found in {latest_version}.")
    
    checkpoints = [
        os.path.join(checkpoints_dir, ckpt)
        for ckpt in os.listdir(checkpoints_dir)
        if ckpt.endswith(".ckpt")
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoints_dir}.")
    
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint


if __name__ == "__main__":
    # Get the latest checkpoint path
    checkpoint_path = get_latest_checkpoint()

    # Load the model from the latest checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = Model.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # Load the test dataset
    testset = Dataset.load_from_disk("data/processed/validation")
    testloader = DataLoader(testset, num_workers=8)

    # Configure the trainer
    if torch.cuda.is_available():
        trainer = pl.Trainer(accelerator="gpu", devices=1)
    else:
        trainer = pl.Trainer()

    # Evaluate the model
    results = trainer.test(model=model, dataloaders=testloader, verbose=True)

    print("Evaluation results:", results)
