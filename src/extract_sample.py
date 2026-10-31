"""Audio sample extractor for project HTML demo page.

Loads a small number of samples from LibriSpeech, applies TacticalAudioDegrader
(bandpass biquad filter + Gaussian noise), and saves degraded audio as .wav files
for use in the index.html Examples section audio players.
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf

# Add project root to path so `src.*` imports work when running this script directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_speech_dataset
from src.preprocess import TacticalAudioDegrader


def extract_samples(
    num_samples: int = 1,
    output_dir: str = ".",
    dataset_name: str = "openslr/librispeech_asr",
    subset: str = "clean",
    split_prefix: str = "validation",
    sample_rate: int = 16000,
    min_snr_db: float = -5.0,
    max_snr_db: float = 0.0,
) -> None:
    """Extract and save tactically degraded audio samples as WAV files.

    Args:
        num_samples: Number of samples to extract.
        output_dir: Directory where WAV files will be saved.
        dataset_name: HF Hub dataset name.
        subset: Dataset subset (e.g. "clean").
        split_prefix: Base split name (e.g. "validation").
        sample_rate: Target audio sample rate in Hz.
        min_snr_db: Minimum SNR for Gaussian noise (dB).
        max_snr_db: Maximum SNR for Gaussian noise (dB).
    """
    # 1. Load a small slice of the dataset
    split = f"{split_prefix}[:{num_samples}]"
    print(f"[INFO] Loading {num_samples} sample(s) from {dataset_name} ({subset}) [{split}]...")
    dataset = load_speech_dataset(
        dataset_name=dataset_name,
        subset=subset,
        split=split,
    )

    # 2. Initialise TacticalAudioDegrader
    degrader = TacticalAudioDegrader(
        min_snr_db=min_snr_db,
        max_snr_db=max_snr_db,
    )

    # 3. Process and save each sample
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(len(dataset)):
        sample = dataset[idx]

        # --- Safely read audio array & sample rate (AudioDecoder-compatible) ---
        audio_data = sample["audio"]
        try:
            audio_array = audio_data["array"]
        except (KeyError, AttributeError, TypeError):
            audio_array = getattr(audio_data, "array", None)

        try:
            sr = audio_data["sampling_rate"]
        except (KeyError, AttributeError, TypeError):
            sr = getattr(audio_data, "sampling_rate", sample_rate)

        if audio_array is None:
            print(f"[WARN] Sample {idx}: could not read audio array. Skipping.")
            continue

        audio_array = np.array(audio_array, dtype=np.float32)

        # --- Apply tactical degradation ---
        degraded_audio = degrader.degrade_audio(audio_array, sr)

        # --- Save as WAV ---
        filename = f"tactical_sample_{idx + 1:02d}.wav"
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, degraded_audio, sr, subtype="PCM_16")

        # --- Print ground-truth text ---
        text = sample.get("text", "(no transcription)")
        print(f"[OK] Saved: {filepath}")
        print(f"     Ground-truth: {text}")

    print(f"\n[DONE] {len(dataset)} sample(s) extracted to '{output_dir}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract tactically degraded audio samples for the HTML demo page."
    )
    parser.add_argument(
        "--num_samples", type=int, default=1,
        help="Number of samples to extract (default: 1).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=".",
        help="Directory to save output WAV files (default: project root '.').",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="openslr/librispeech_asr",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--subset", type=str, default="clean",
        help="Dataset subset (default: 'clean').",
    )
    parser.add_argument(
        "--split_prefix", type=str, default="validation",
        help="Dataset split prefix (default: 'validation').",
    )
    parser.add_argument(
        "--min_snr_db", type=float, default=-5.0,
        help="Minimum SNR for Gaussian noise in dB (default: -5.0).",
    )
    parser.add_argument(
        "--max_snr_db", type=float, default=0.0,
        help="Maximum SNR for Gaussian noise in dB (default: 0.0).",
    )
    args = parser.parse_args()

    extract_samples(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        subset=args.subset,
        split_prefix=args.split_prefix,
        min_snr_db=args.min_snr_db,
        max_snr_db=args.max_snr_db,
    )


if __name__ == "__main__":
    main()

