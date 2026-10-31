"""Data loader module for speech dataset downloading and initial loading."""

import os
from typing import List, Optional, Union
from datasets import Dataset, load_dataset


def load_speech_dataset(
    dataset_name: str = "openslr/librispeech_asr",
    subset: Optional[str] = "clean",
    split: Optional[str] = "validation[:50]",
    data_files: Optional[Union[str, List[str]]] = None,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """Download and load speech dataset from Hugging Face Datasets or local files.

    Supports Hugging Face hub datasets as well as local parquet/arrow/csv/json files.

    Args:
        dataset_name: Name of dataset on HF Hub or dataset format (e.g., 'parquet').
        subset: Dataset configuration or subset name (default: "clean").
        split: Dataset split to load (e.g. "validation[:50]", "train", None).
        data_files: Path or pattern to local data files (e.g. "data/raw/*.parquet").
        cache_dir: Optional directory to cache downloaded data.

    Returns:
        Dataset: Loaded Hugging Face Dataset instance.
    """
    # If data_files is provided, load local files directly
    if data_files is not None:
        print(f"Loading local dataset files from: {data_files}...")
        fmt = "parquet" if (isinstance(data_files, str) and data_files.endswith(".parquet")) else dataset_name
        dataset = load_dataset(fmt, data_files=data_files, split=split or "train", cache_dir=cache_dir)
        print(f"Local dataset successfully loaded. Number of samples: {len(dataset)}")
        return dataset

    # Check if dataset_name is a local path containing parquet files
    if os.path.exists(dataset_name) and (dataset_name.endswith(".parquet") or os.path.isdir(dataset_name)):
        print(f"Loading local dataset from path: {dataset_name}...")
        dataset = load_dataset("parquet", data_files=dataset_name, split=split or "train", cache_dir=cache_dir)
        print(f"Local dataset successfully loaded. Number of samples: {len(dataset)}")
        return dataset

    print(f"Loading dataset from HF Hub: {dataset_name} ({subset}) [{split}]...")
    dataset = load_dataset(dataset_name, subset, split=split, cache_dir=cache_dir)
    print(f"Dataset successfully loaded. Number of samples: {len(dataset)}")
    return dataset


if __name__ == "__main__":
    # Test downloading and loading dataset
    sample_dataset = load_speech_dataset()
    print("First sample keys:", sample_dataset[0].keys())
    print("Sample text:", sample_dataset[0]["text"])
