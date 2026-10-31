"""Main training and evaluation pipeline for tactical speech recognition.

Integrates data loading, degradation, preprocessing, LoRA modeling, and training.
Supports YAML configuration and reproducible experiment tracking.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path when running script directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.utils import load_yaml_config, set_seed, setup_logger


def evaluate_wer(
    model: Any,
    processor: Any,
    dataset: Any,
    device: Optional[Any] = None,
    metric_name: str = "wer",
    desc: str = "Inference & Evaluation",
    logger: Optional[Any] = None,
) -> Tuple[float, List[str], List[str]]:
    """Evaluate Speech-to-Text model on dataset using Word Error Rate (WER)."""
    import evaluate
    import torch
    from tqdm.auto import tqdm
    from src.model import get_default_device

    if device is None:
        device = get_default_device()

    log = logger.info if logger else print

    log(f"Loading evaluation metric: {metric_name}...")
    wer_metric = evaluate.load(metric_name)

    model.eval()
    predictions: List[str] = []
    references: List[str] = []

    log(f"Starting inference on {len(dataset)} samples ({desc})...")
    for i in tqdm(range(len(dataset)), desc=desc):
        sample = dataset[i]
        audio = sample["audio"]
        ground_truth = sample["text"]

        inputs = processor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        )
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        predictions.append(transcription.strip().upper())
        references.append(ground_truth.strip().upper())

    wer_score = wer_metric.compute(predictions=predictions, references=references)
    return float(wer_score), predictions, references


def train_whisper(
    model: Any,
    train_dataset: Any,
    data_collator: Any,
    output_dir: str = "./checkpoints/tactical_whisper",
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    num_train_epochs: int = 2,
    logging_steps: int = 5,
    fp16: Optional[bool] = None,
    logger: Optional[Any] = None,
) -> Any:
    """Configure Seq2SeqTrainer and run LoRA fine-tuning for Whisper."""
    import torch
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    log = logger.info if logger else print

    if fp16 is None:
        fp16 = torch.cuda.is_available()

    log("Configuring Seq2SeqTrainingArguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        remove_unused_columns=False,
        label_names=["labels"],
        fp16=fp16,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    log(f"Starting model training (Epochs: {num_train_epochs}, Batch Size: {batch_size})...")
    trainer.train()
    log("Model fine-tuning finished successfully!")
    return trainer


def run_pipeline(
    model_id: str = "openai/whisper-small",
    dataset_name: str = "openslr/librispeech_asr",
    subset: str = "clean",
    split: str = "validation[:50]",
    output_dir: str = "./checkpoints/tactical_whisper",
    epochs: int = 2,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    seed: int = 42,
    skip_zero_shot: bool = False,
    logger: Optional[Any] = None,
) -> Dict[str, float]:
    """Run full end-to-end tactical speech recognition training and evaluation pipeline."""
    if logger is None:
        logger = setup_logger("train_pipeline")

    # Set random seed for reproducibility
    set_seed(seed)

    from src.data_loader import load_speech_dataset
    from src.model import get_default_device, load_whisper_model_and_processor, setup_lora_model
    from src.preprocess import (
        DataCollatorSpeechSeq2SeqWithPadding,
        TacticalAudioDegrader,
        prepare_dataset_for_whisper,
    )

    device = get_default_device()
    logger.info(f"Executing pipeline on device: {device}")

    # 1. Download and load raw clean dataset
    raw_dataset = load_speech_dataset(dataset_name=dataset_name, subset=subset, split=split)

    # 2. Apply tactical battlefield degradation (narrowband radio + Gaussian noise)
    degrader = TacticalAudioDegrader(min_snr_db=-5.0, max_snr_db=0.0, central_freq=1400.0, q=0.707)
    tactical_dataset = degrader.transform_dataset(raw_dataset)

    # 3. Load base model and processor
    model, processor = load_whisper_model_and_processor(model_id=model_id, device=device)

    # 4. Zero-shot baseline evaluation
    zero_shot_wer = None
    if not skip_zero_shot:
        logger.info("\n--- [Step 1] Zero-shot Baseline Evaluation ---")
        zero_shot_wer, zero_preds, zero_refs = evaluate_wer(
            model=model,
            processor=processor,
            dataset=tactical_dataset,
            device=device,
            desc="Zero-shot Evaluation",
            logger=logger,
        )
        logger.info("========================================")
        logger.info(f"🔥 Extreme Tactical Zero-shot WER: {zero_shot_wer * 100:.2f}%")
        logger.info("========================================")
        if zero_refs and zero_preds:
            logger.info(f"[Reference]:  {zero_refs[0]}")
            logger.info(f"[Prediction]: {zero_preds[0]}")

    # 5. Preprocess dataset for Whisper (Mel-spectrogram & Token IDs)
    logger.info("\n--- [Step 2] Dataset Preprocessing for Whisper Training ---")
    prepared_dataset = prepare_dataset_for_whisper(tactical_dataset, processor=processor)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 6. LoRA Model Setup
    logger.info("\n--- [Step 3] LoRA Setup ---")
    peft_model = setup_lora_model(
        model=model,
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    # 7. Model Fine-Tuning
    logger.info("\n--- [Step 4] LoRA Fine-Tuning ---")
    train_whisper(
        model=peft_model,
        train_dataset=prepared_dataset,
        data_collator=data_collator,
        output_dir=output_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        logger=logger,
    )

    # 8. Post Fine-Tuning Evaluation
    logger.info("\n--- [Step 5] Post Fine-Tuning Evaluation ---")
    ft_wer, ft_preds, ft_refs = evaluate_wer(
        model=peft_model,
        processor=processor,
        dataset=tactical_dataset,
        device=device,
        desc="Post Fine-Tuning Evaluation",
        logger=logger,
    )
    logger.info("========================================")
    logger.info(f"✅ Post Fine-Tuning WER: {ft_wer * 100:.2f}%")
    if zero_shot_wer is not None:
        wer_diff = zero_shot_wer - ft_wer
        logger.info(f"📈 WER Improvement: {wer_diff * 100:+.2f}%p (Absolute)")
    logger.info("========================================")
    if ft_refs and ft_preds:
        logger.info(f"[Reference]:  {ft_refs[0]}")
        logger.info(f"[FT Prediction]: {ft_preds[0]}")

    results = {"post_ft_wer": ft_wer}
    if zero_shot_wer is not None:
        results["zero_shot_wer"] = zero_shot_wer
    return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments with optional YAML config support."""
    parser = argparse.ArgumentParser(
        description="Speech Challenge 2026 - Tactical Whisper Training Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (e.g., configs/default_config.yaml)",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Hugging Face model ID (default: openai/whisper-small)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name (default: openslr/librispeech_asr)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Dataset configuration/subset (default: clean)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split (default: validation[:50])",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Training output directory (default: ./checkpoints/tactical_whisper)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: 2)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size per device (default: 4)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--skip_zero_shot",
        action="store_true",
        default=None,
        help="Skip initial zero-shot baseline evaluation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()

    # Load YAML config if specified or fallback to configs/default_config.yaml if present
    cfg = {}
    config_file = cli_args.config
    if config_file and os.path.exists(config_file):
        cfg = load_yaml_config(config_file)
    elif os.path.exists("configs/default_config.yaml"):
        cfg = load_yaml_config("configs/default_config.yaml")

    # Merge config file and CLI arguments (CLI takes precedence)
    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("training", {})
    exp_cfg = cfg.get("experiment", {})

    model_id = cli_args.model_id or model_cfg.get("model_id", "openai/whisper-small")
    dataset_name = cli_args.dataset_name or dataset_cfg.get("name", "openslr/librispeech_asr")
    subset = cli_args.subset or dataset_cfg.get("subset", "clean")
    split = cli_args.split or dataset_cfg.get("train_split", "validation[:50]")
    output_dir = cli_args.output_dir or train_cfg.get("output_dir", "./checkpoints/tactical_whisper")
    epochs = cli_args.epochs or train_cfg.get("num_train_epochs", 2)
    batch_size = cli_args.batch_size or train_cfg.get("per_device_train_batch_size", 4)
    learning_rate = cli_args.learning_rate or train_cfg.get("learning_rate", 1e-3)
    seed = cli_args.seed or exp_cfg.get("seed", 42)
    skip_zero_shot = cli_args.skip_zero_shot if cli_args.skip_zero_shot is not None else train_cfg.get("skip_zero_shot", False)

    logger = setup_logger("train")
    if config_file:
        logger.info(f"Loaded configuration from: {config_file}")

    run_pipeline(
        model_id=model_id,
        dataset_name=dataset_name,
        subset=subset,
        split=split,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        skip_zero_shot=skip_zero_shot,
        logger=logger,
    )
