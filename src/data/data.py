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
    # Set raw_dir to the "data/raw" directory relative to the project root
    raw_dir = Path(__file__).resolve().parents[2] / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download the WMT19 dataset
    print("Downloading the WMT19 English-to-German dataset...")
    dataset = load_dataset("wmt19", "de-en", cache_dir=str(raw_dir))

    # Reduce the size of the dataset
    print(f"Reducing the dataset to a maximum of {max_samples} samples per split...")
    reduced_dataset = {}
    for split, data in dataset.items():
        reduced_data = data.select(range(min(len(data), max_samples)))
        reduced_dataset[split] = reduced_data

    # Save the reduced dataset to the raw_dir
    for split, data in reduced_dataset.items():
        save_path = raw_dir / f"{split}.arrow"
        print(f"Saving reduced {split} data to {save_path}")
        data.save_to_disk(str(save_path))

    print(f"Reduced dataset has been saved to {raw_dir}.")

if __name__ == "__main__":
    download_and_reduce_wmt19(max_samples=50000)
