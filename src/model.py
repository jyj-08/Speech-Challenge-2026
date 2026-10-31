"""Transformer model loading and PEFT/LoRA configuration module."""

from typing import List, Optional, Tuple, Union
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def get_default_device() -> torch.device:
    """Return cuda device if available, else cpu."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_whisper_model_and_processor(
    model_id: str = "openai/whisper-small",
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """Load pretrained Whisper model and processor.

    Args:
        model_id: Hugging Face model identifier (default: "openai/whisper-small").
        device: Target torch device. If None, auto-detects CUDA/CPU.

    Returns:
        Tuple of (model, processor).
    """
    if device is None:
        device = get_default_device()
    else:
        device = torch.device(device)

    print(f"Loading Whisper model and processor from '{model_id}'...")
    print(f"Target device: {device}")

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)

    return model, processor


def setup_lora_model(
    model: WhisperForConditionalGeneration,
    r: int = 8,
    lora_alpha: int = 32,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.05,
    bias: str = "none",
) -> PeftModel:
    """Wrap Whisper model with LoRA (Low-Rank Adaptation) using PEFT.

    Args:
        model: Base Whisper model.
        r: LoRA rank dimension.
        lora_alpha: LoRA scaling factor.
        target_modules: Module names to apply LoRA (defaults to ["q_proj", "v_proj"]).
        lora_dropout: Dropout probability for LoRA layers.
        bias: Bias training configuration ("none", "all", "lora_only").

    Returns:
        PeftModel: LoRA-wrapped trainable model.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    print("Configuring LoRA (Parameter-Efficient Fine-Tuning)...")
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model
