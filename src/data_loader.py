"""Data loader module for speech dataset downloading and initial loading."""

from typing import Optional
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset


def load_speech_dataset(
    dataset_name: str = "openslr/librispeech_asr",
    subset: str = "clean",
    split: str = "validation[:50]",
    cache_dir: Optional[str] = None,
) -> Dataset:
    """Download and load speech dataset from Hugging Face Datasets.

    Args:
        dataset_name: Name of the dataset on Hugging Face Hub (default: "openslr/librispeech_asr").
        subset: Dataset configuration or subset name (default: "clean").
        split: Dataset split to load (e.g. "validation[:50]", "train", "test").
        cache_dir: Optional directory to cache downloaded data.

    Returns:
        Dataset: Loaded Hugging Face Dataset instance.
    """
    print(f"Loading dataset: {dataset_name} ({subset}) [{split}]...")
    dataset = load_dataset(dataset_name, subset, split=split, cache_dir=cache_dir)
    print(f"Dataset successfully loaded. Number of samples: {len(dataset)}")
    return dataset


if __name__ == "__main__":
    # Test downloading and loading dataset
    sample_dataset = load_speech_dataset()
    print("First sample keys:", sample_dataset[0].keys())
    print("Sample text:", sample_dataset[0]["text"])

