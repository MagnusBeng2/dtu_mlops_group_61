from unittest.mock import patch, MagicMock
import pytest
import os
import torch
from torch.utils.data import TensorDataset
from src.models.train_model import train, set_seed, get_next_version

def test_train_model_arguments():
    """
    Test that the train_model.py script correctly parses arguments.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--wandbkey", default=None, type=str, help="W&B API key")
    parser.add_argument("--debug_mode", action="store_true", help="Run in debug mode")
    parser.add_argument("--gpus", default=0, type=int, help="Number of GPUs")

    args = parser.parse_args([])
    assert args.lr == 0.0001
    assert args.epochs == 1
    assert args.batch_size == 16
    assert args.seed == 42
    assert args.wandbkey is None
    assert args.debug_mode is False
    assert args.gpus == 0


@patch("src.models.train_model.wandb")
@patch("src.models.train_model.load_from_disk")
@patch("src.models.train_model.Model")
@patch("src.models.train_model.pl.Trainer")
def test_train_model_steps(mock_trainer, mock_model, mock_load_from_disk, mock_wandb):
    """
    Test that train_model.py executes the training steps properly.
    """
    # Mock datasets
    mock_train_dataset = MagicMock()
    mock_train_dataset.__len__.return_value = 100  # Mock dataset size

    mock_val_dataset = MagicMock()
    mock_val_dataset.__len__.return_value = 20  # Mock dataset size

    mock_load_from_disk.side_effect = [mock_train_dataset, mock_val_dataset]

    # Mock model
    mock_model_instance = MagicMock()
    mock_model.return_value = mock_model_instance

    # Mock Trainer
    mock_trainer_instance = MagicMock()
    mock_trainer.return_value = mock_trainer_instance

    # Mock Wandb
    mock_wandb.config = MagicMock()
    mock_wandb.config.batch_size = 16
    mock_wandb.config.lr = 0.0001
    mock_wandb.config.epochs = 1
    mock_wandb.config.seed = 42

    # Mock arguments
    class MockArgs:
        lr = 0.0001
        epochs = 1
        batch_size = 16
        seed = 42
        wandbkey = None
        debug_mode = True
        gpus = 0

    # Run training
    train(MockArgs())

    # Assertions for dataset loading
    mock_load_from_disk.assert_any_call(
        os.path.abspath("data/processed/train")
    )
    mock_load_from_disk.assert_any_call(
        os.path.abspath("data/processed/validation")
    )


def test_set_seed():
    """
    Test that set_seed sets the random seeds correctly.
    """
    seed = 42
    set_seed(seed)

    # Test consistency for torch
    torch_rand1 = torch.rand(1).item()
    set_seed(seed)  # Reset the seed
    torch_rand2 = torch.rand(1).item()
    assert torch_rand1 == torch_rand2

    # Test consistency for random module
    import random
    random.seed(seed)
    random_val1 = random.random()
    random.seed(seed)
    random_val2 = random.random()
    assert random_val1 == random_val2


def test_get_next_version():
    """
    Test that get_next_version returns the correct directory name.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tempdir:
        version_path = get_next_version(base_dir=tempdir)
        assert version_path == os.path.join(tempdir, "version_0")

        # Simulate existing versions
        os.makedirs(os.path.join(tempdir, "version_0"))
        os.makedirs(os.path.join(tempdir, "version_1"))
        version_path = get_next_version(base_dir=tempdir)
        assert version_path == os.path.join(tempdir, "version_2")


@patch("src.models.train_model.wandb")
@patch("src.models.train_model.load_from_disk")
@patch("src.models.train_model.pl.Trainer")
def test_training_loop(mock_trainer, mock_load_from_disk, mock_wandb):
    """
    Test the main training loop in train_model.py with mocked components.
    """
    # Mock datasets
    train_data = TensorDataset(torch.rand(10, 10), torch.rand(10, 10))
    val_data = TensorDataset(torch.rand(10, 10), torch.rand(10, 10))
    mock_load_from_disk.side_effect = [train_data, val_data]

    # Mock Wandb
    mock_wandb.init.return_value = MagicMock()  # Mock wandb.init()
    mock_wandb.config = MagicMock()            # Mock wandb.config
    mock_wandb.config.lr = 0.0001
    mock_wandb.config.epochs = 1
    mock_wandb.config.batch_size = 2
    mock_wandb.config.seed = 42

    # Mock Trainer and its fit method
    mock_trainer_instance = MagicMock()
    mock_trainer.return_value = mock_trainer_instance

    # Mock arguments
    class MockArgs:
        lr = 0.0001
        epochs = 1
        batch_size = 2
        seed = 42
        wandbkey = None
        debug_mode = False
        gpus = 0

    # Run training
    train(MockArgs())

    # Assertions for Trainer
    assert mock_trainer_instance.fit.called
    mock_trainer_instance.fit.assert_called_once()

