from pathlib import Path
from datasets import load_dataset


def download_raw_data() -> None:
    """
    Downloads the IWSLT2017 English-to-German dataset and saves it to /data/raw.

    Returns:
        None
    """
    # Set the raw data directory to /data/raw in the root directory
    raw_dir = Path(__file__).resolve().parents[2] / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if the required files already exist
    expected_files = ["train.arrow", "validation.arrow", "test.arrow"]
    if all((raw_dir / file).exists() for file in expected_files):
        print(f"Raw data already exists in {raw_dir}. Skipping download.")
        return

    # Download the dataset with the correct language pair and trust remote code
    print(f"Downloading IWSLT2017 English-to-German dataset into {raw_dir}...")
    dataset = load_dataset("iwslt2017", "iwslt2017-en-de", cache_dir=str(raw_dir), trust_remote_code=True)

    # Save the dataset splits to the raw directory
    for split, data in dataset.items():
        save_path = raw_dir / f"{split}.arrow"
        print(f"Saving {split} data to {save_path}")
        data.save_to_disk(str(save_path))

    print(f"Raw data has been downloaded and saved to {raw_dir}.")


if __name__ == "__main__":
    download_raw_data()
