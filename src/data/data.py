from pathlib import Path
from datasets import load_dataset

def download_and_save_wmt19() -> None:
    """
    Downloads the WMT19 English-to-German dataset and saves the complete dataset splits as .arrow files.

    Returns:
        None
    """
    raw_dir = Path(__file__).resolve().parents[2] / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_dir = raw_dir / "wmt19"  # Updated to match make_dataset.py
    save_dir.mkdir(exist_ok=True)

    # Check if the dataset splits already exist
    if all((save_dir / f"{split}.arrow").exists() for split in ["train", "validation"]):
        print(f"Complete dataset already exists at {save_dir}. Skipping download.")
        return

    print("Downloading the WMT19 English-to-German dataset...")
    dataset = load_dataset("wmt19", "de-en", cache_dir=str(raw_dir))

    print("Saving the complete dataset splits as .arrow files...")
    for split in ["train", "validation"]:
        if split in dataset:
            save_path = save_dir / f"{split}.arrow"
            print(f"Saving {split} data to {save_path}")
            dataset[split].save_to_disk(str(save_path))

    print(f"Complete dataset has been saved to {save_dir} as .arrow files.")


if __name__ == "__main__":
    download_and_save_wmt19()
