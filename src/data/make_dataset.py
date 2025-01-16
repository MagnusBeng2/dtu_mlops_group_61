# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
from typing import Optional

import click
from datasets import Dataset, load_from_disk
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument("raw_dir", type=click.Path(exists=True))
@click.argument("processed_dir", type=click.Path())
@click.option("--k", default=None, type=int, help="Number of samples to include in the processed dataset")
def main(raw_dir: str, processed_dir: str, k: Optional[int] = None) -> None:
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).

    Parameters
    ----------
    raw_dir : str, required
        A path to the raw data directory.
    processed_dir : str, required
        A path to save the processed data.
    k : integer, optional
        The amount of datapoints to include from the dataset.

    Raises
    ------
    TypeError
        If raw_dir or processed_dir aren't strings.
    ValueError
        If k is a negative integer.
    """
    if type(raw_dir) is not str or type(processed_dir) is not str:
        raise TypeError("Both raw_dir and processed_dir must be strings denoting paths.")
    if k is not None and k <= 0:
        raise ValueError("k must be a positive amount of datapoints.")

    logger = logging.getLogger(__name__)
    logger.info("Processing data from raw to processed format...")

    # Ensure processed_dir exists
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data from raw_dir
    logger.info(f"Loading raw data from {raw_dir}...")
    raw_dir = Path(raw_dir)
    train_data = load_from_disk(raw_dir / "train.arrow")
    val_data = load_from_disk(raw_dir / "validation.arrow")

    # Limit the dataset size if k is specified
    if k is not None:
        logger.info(f"Sampling up to {k} datapoints from each dataset split...")
        train_data = train_data.select(range(min(len(train_data), k)))
        val_data = val_data.select(range(min(len(val_data), k)))

    # Example preprocessing: lowercasing and stripping text
    def preprocess_function(examples):
        return {
            "translation": [
                {
                    "en": translation["en"].strip().lower(),
                    "de": translation["de"].strip().lower(),
                }
                for translation in examples["translation"]
            ]
        }

    # Apply preprocessing
    print("Preprocessing the dataset...")
    train_data = train_data.map(preprocess_function, batched=True)
    val_data = val_data.map(preprocess_function, batched=True)

    logger.info("Preprocessing the dataset...")
    train_data = train_data.map(preprocess_function, batched=True)
    val_data = val_data.map(preprocess_function, batched=True)

    # Save processed data to the processed_dir
    logger.info(f"Saving processed data to {processed_dir}...")
    train_data.save_to_disk(processed_dir / "train")
    val_data.save_to_disk(processed_dir / "validation")

    logger.info("Data processing complete.")


if __name__ == "__main__":
    # Setup logging
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Load environment variables if needed
    load_dotenv(find_dotenv())

    # Run the CLI
    main()
