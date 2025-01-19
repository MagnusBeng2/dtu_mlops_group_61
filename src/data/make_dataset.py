# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import Optional

import click
from datasets import load_from_disk
from dotenv import find_dotenv, load_dotenv
from transformers import T5Tokenizer


@click.command()
@click.option("--k", default=None, type=int, help="Number of samples to include in the processed dataset")
def main(k: Optional[int] = None) -> None:
    """
    Runs data processing scripts to turn raw data from /data/raw/reduced_50000 into
    cleaned data ready to be analyzed (saved in /data/processed).

    Parameters
    ----------
    k : integer, optional
        The number of datapoints to include in the processed dataset.

    Raises
    ------
    ValueError
        If k is a negative integer.
    """
    # Define default directories
    raw_dir = Path(__file__).resolve().parents[2] / "data/raw/reduced_50000"
    processed_dir = Path(__file__).resolve().parents[2] / "data/processed"

    # Validate k
    if k is not None and k <= 0:
        raise ValueError("k must be a positive number of datapoints.")

    logger = logging.getLogger(__name__)
    logger.info("Processing data from /data/raw/reduced_50000 to /data/processed...")

    # Ensure processed_dir exists
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Convert raw_dir to string for compatibility with `load_from_disk`
    logger.info(f"Loading raw data from {raw_dir}...")
    train_data = load_from_disk(str(raw_dir / "train.arrow"))
    val_data = load_from_disk(str(raw_dir / "validation.arrow"))

    # Limit the dataset size if k is specified
    if k is not None:
        logger.info(f"Sampling up to {k} datapoints from each dataset split...")
        train_data = train_data.select(range(min(len(train_data), k)))
        val_data = val_data.select(range(min(len(val_data), k)))

    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Define preprocessing function
    def preprocess_function(examples):
        model_inputs = tokenizer(
            [translation["en"] for translation in examples["translation"]],
            max_length=128,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            [translation["de"] for translation in examples["translation"]],
            max_length=128,
            padding="max_length",
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Apply preprocessing
    logger.info("Preprocessing the dataset...")
    print(f"Train data columns: {train_data.column_names}")
    print(f"Validation data columns: {val_data.column_names}")
    train_data = train_data.map(preprocess_function, batched=True)
    val_data = val_data.map(preprocess_function, batched=True)

    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Save processed data to the processed_dir
    logger.info(f"Saving processed data to {processed_dir}...")
    train_data.save_to_disk(str(processed_dir / "train"))
    val_data.save_to_disk(str(processed_dir / "validation"))

    logger.info("Data processing complete.")


if __name__ == "__main__":
    # Setup logging
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Load environment variables if needed
    load_dotenv(find_dotenv())

    # Run the CLI
    main()
