"""Audio preprocessing and augmentation pipeline module.

Includes tactical radio degradation simulation (narrowband filtering + Gaussian noise)
and Hugging Face Whisper feature extraction and batch collation.
Robust against various audio data structures including dict and torchcodec AudioDecoder.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torchaudio
from audiomentations import Compose, AddGaussianSNR
from datasets import Dataset


class TacticalAudioDegrader:
    """Simulates extreme tactical battlefield communication audio degradation.

    Applies:
    1. Bandpass biquad filter (narrowband radio simulation, e.g., 300Hz ~ 2500Hz).
    2. Additive Gaussian SNR noise via audiomentations.
    """

    def __init__(
        self,
        min_snr_db: float = -5.0,
        max_snr_db: float = 0.0,
        central_freq: float = 1400.0,
        q: float = 0.707,
    ):
        self.central_freq = central_freq
        self.q = q
        self.pipeline = Compose([
            AddGaussianSNR(min_snr_db=min_snr_db, max_snr_db=max_snr_db, p=1.0)
        ])

    def degrade_audio(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply tactical degradation to a single audio numpy array."""
        # Step 1: Bandpass biquad filtering for narrowband radio simulation
        tensor_audio = torch.tensor(audio_array).unsqueeze(0).to(torch.float32)
        narrowband_audio = torchaudio.functional.bandpass_biquad(
            tensor_audio, sample_rate=sample_rate, central_freq=self.central_freq, Q=self.q
        )

        # Step 2: Gaussian noise mixup
        narrowband_np = narrowband_audio.squeeze().cpu().numpy()
        noisy_audio = self.pipeline(samples=narrowband_np, sample_rate=sample_rate)
        return noisy_audio

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply degradation to a single batch dict from a Hugging Face Dataset.

        Safely handles both dictionary and custom object audio representations
        (e.g., torchcodec AudioDecoder) without relying on dict-only methods like .get().
        """
        audio_data = batch["audio"]

        try:
            audio_array = audio_data["array"]
        except (KeyError, AttributeError, TypeError):
            audio_array = getattr(audio_data, "array", None)

        try:
            sample_rate = audio_data["sampling_rate"]
        except (KeyError, AttributeError, TypeError):
            sample_rate = getattr(audio_data, "sampling_rate", 16000)

        # Safely access audio path without using dict-only .get()
        try:
            audio_path = audio_data["path"]
        except (KeyError, AttributeError, TypeError):
            audio_path = ""

        noisy_audio = self.degrade_audio(audio_array, sample_rate)

        batch["audio"] = {
            "array": noisy_audio,
            "sampling_rate": sample_rate,
            "path": audio_path,
        }
        return batch

    def transform_dataset(self, dataset: Dataset, num_proc: Optional[int] = None) -> Dataset:
        """Map degradation over an entire dataset."""
        print("Applying tactical degradation (narrowband filter + noise) to dataset...")
        degraded = dataset.map(self.__call__, num_proc=num_proc)
        print("Tactical dataset construction completed!")
        return degraded


def prepare_dataset_for_whisper(
    dataset: Dataset,
    processor: Any,
    remove_columns: bool = True,
    num_proc: Optional[int] = None,
) -> Dataset:
    """Preprocess dataset for Whisper: converts audio to log-mel spectrogram and text to token IDs.

    Args:
        dataset: Input audio dataset.
        processor: WhisperProcessor instance.
        remove_columns: Whether to remove original dataset columns.
        num_proc: Number of processes for parallel mapping.

    Returns:
        Dataset: Processed dataset ready for Seq2SeqTrainer.
    """
    def _prepare(batch):
        audio = batch["audio"]

        try:
            audio_array = audio["array"]
        except (KeyError, AttributeError, TypeError):
            audio_array = getattr(audio, "array", None)

        try:
            sample_rate = audio["sampling_rate"]
        except (KeyError, AttributeError, TypeError):
            sample_rate = getattr(audio, "sampling_rate", 16000)

        batch["input_features"] = processor.feature_extractor(
            audio_array, sampling_rate=sample_rate
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    columns_to_remove = dataset.column_names if remove_columns else None
    print("Preparing dataset (Mel-spectrogram & Token IDs) for Whisper...")
    prepared = dataset.map(_prepare, remove_columns=columns_to_remove, num_proc=num_proc)
    return prepared


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for Whisper Speech-to-Text Seq2Seq models.

    Handles different lengths of input log-Mel spectrograms and output token sequences,
    padding them dynamically to batch maximum length and masking padded tokens with -100.
    """

    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Mask padding in labels with -100 so it is ignored by CrossEntropyLoss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is prepended in both encoder and decoder, strip leading bos
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
