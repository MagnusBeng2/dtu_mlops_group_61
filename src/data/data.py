from pathlib import Path
from datasets import load_dataset

def download_and_reduce_wmt19(max_samples: int = 50000) -> None:
    """
    Downloads the WMT19 English-to-German dataset and reduces its size to be under 500 MB.

    Args:
        max_samples (int): Maximum number of samples to keep in each dataset split.

    Returns:
        None
    """
    raw_dir = Path(__file__).resolve().parents[2] / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_dir = raw_dir / f"reduced_{max_samples}"
    save_dir.mkdir(exist_ok=True)

    if all((save_dir / f"{split}.arrow").exists() for split in ["train", "validation", "test"]):
        print(f"Reduced dataset already exists at {save_dir}. Skipping download.")
        return

    print("Downloading the WMT19 English-to-German dataset...")
    dataset = load_dataset("wmt19", "de-en", cache_dir=str(raw_dir))

    print(f"Reducing the dataset to a maximum of {max_samples} samples per split...")
    reduced_dataset = {}
    for split, data in dataset.items():
        reduced_data = data.select(range(min(len(data), max_samples)))
        reduced_dataset[split] = reduced_data

    for split, data in reduced_dataset.items():
        save_path = save_dir / f"{split}.arrow"
        print(f"Saving reduced {split} data to {save_path}")
        data.save_to_disk(str(save_path))

    print(f"Reduced dataset has been saved to {save_dir}.")


if __name__ == "__main__":
    download_and_reduce_wmt19(max_samples=50000)