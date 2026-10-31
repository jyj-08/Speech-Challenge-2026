"""Unit tests for audio preprocessing and degradation pipelines."""

import os
import sys
import numpy as np
import pytest

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.preprocess import TacticalAudioDegrader


@pytest.fixture
def sample_audio_data():
    """Generate dummy audio data fixture (1 second of 16kHz sine wave)."""
    sample_rate = 16000
    duration_sec = 1.0
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Sine wave of 440Hz
    audio_array = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return {
        "audio": {
            "array": audio_array,
            "sampling_rate": sample_rate,
            "path": "dummy_path.wav",
        },
        "text": "THIS IS A TEST AUDIO FOR DEGRADATION",
    }


def test_tactical_audio_degrader_dimensions(sample_audio_data):
    """Test that degrade_audio preserves audio length and 1D dimension without tensor errors."""
    degrader = TacticalAudioDegrader(
        min_snr_db=-5.0,
        max_snr_db=0.0,
        central_freq=1400.0,
        q=0.707,
    )

    audio_array = sample_audio_data["audio"]["array"]
    sample_rate = sample_audio_data["audio"]["sampling_rate"]

    degraded_audio = degrader.degrade_audio(audio_array, sample_rate)

    # 1. Dimension and shape check
    assert isinstance(degraded_audio, np.ndarray), "Output must be a numpy ndarray"
    assert degraded_audio.ndim == 1, f"Expected 1D audio array, got {degraded_audio.ndim}D"
    assert degraded_audio.shape == audio_array.shape, (
        f"Shape mismatch: original {audio_array.shape} vs degraded {degraded_audio.shape}"
    )

    # 2. Numerical validity check
    assert not np.isnan(degraded_audio).any(), "Degraded audio contains NaN values"
    assert not np.isinf(degraded_audio).any(), "Degraded audio contains Inf values"


def test_apply_tactical_degradation_batch_dict(sample_audio_data):
    """Test degradation on standard Hugging Face dataset batch dictionary."""
    degrader = TacticalAudioDegrader()

    processed_batch = degrader(sample_audio_data)

    # Dictionary structure preservation check
    assert "audio" in processed_batch, "Missing 'audio' key in processed batch"
    assert "text" in processed_batch, "Missing 'text' key in processed batch"

    audio_info = processed_batch["audio"]
    assert "array" in audio_info, "Missing 'array' key in batch['audio']"
    assert "sampling_rate" in audio_info, "Missing 'sampling_rate' key in batch['audio']"
    assert "path" in audio_info, "Missing 'path' key in batch['audio']"

    # Shape and values validation
    orig_length = len(sample_audio_data["audio"]["array"])
    assert len(audio_info["array"]) == orig_length, "Audio sample count changed after degradation"
    assert audio_info["sampling_rate"] == 16000, "Sampling rate was altered"


@pytest.mark.parametrize("duration_sec", [0.25, 0.5, 2.0])
def test_degrader_various_audio_lengths(duration_sec):
    """Test degradation across various audio lengths (ensures no fixed tensor dimension bounds)."""
    degrader = TacticalAudioDegrader()
    sr = 16000
    n_samples = int(sr * duration_sec)
    dummy_audio = np.random.uniform(-0.8, 0.8, n_samples).astype(np.float32)

    result = degrader.degrade_audio(dummy_audio, sample_rate=sr)
    assert len(result) == n_samples
    assert result.ndim == 1


if __name__ == "__main__":
    # Allow running directly via python tests/test_preprocess.py
    import pytest

    pytest.main([__file__, "-v"])

