"""Evaluation metrics and inference pipeline for tactical speech recognition.

Computes Word Error Rate (WER) and Character Error Rate (CER) on test/validation datasets
for base Whisper models or fine-tuned LoRA checkpoints.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

def load_model_for_eval(base_model_id, checkpoint_dir):
    # 상위 디렉토리(tactical_whisper)가 주어졌을 때 내부의 checkpoint 폴더 자동 스캔
    if os.path.exists(checkpoint_dir):
        subdirs = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) 
                   if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))]
        if subdirs:
            # 파일 생성 시간을 기준으로 가장 최신 폴더 선택
            checkpoint_dir = max(subdirs, key=os.path.getmtime)
            print(f"[INFO] 최신 LoRA 체크포인트 경로 자동 감지: {checkpoint_dir}")

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.utils import setup_logger


def compute_wer_cer(
    predictions: List[str],
    references: List[str],
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute WER (Word Error Rate) and CER (Character Error Rate).

    Args:
        predictions: List of transcribed predicted texts.
        references: List of ground-truth reference texts.
        metrics: List of metric names to evaluate (e.g. ['wer', 'cer']).

    Returns:
        Dict[str, float]: Dictionary containing computed scores (e.g. {'wer': 0.15, 'cer': 0.08}).
    """
    import evaluate

    if metrics is None:
        metrics = ["wer", "cer"]

    results = {}
    for metric_name in metrics:
        try:
            metric = evaluate.load(metric_name)
            score = metric.compute(predictions=predictions, references=references)
            results[metric_name] = float(score)
        except Exception as e:
            print(f"[Warning] Failed to compute metric '{metric_name}': {e}")

    return results


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


def run_evaluation(
    model: Any,
    processor: Any,
    dataset: Any,
    device: Optional[Any] = None,
    metrics: Optional[List[str]] = None,
    logger: Optional[Any] = None,
) -> Tuple[Dict[str, float], List[Dict[str, str]]]:
    """Run inference over dataset and compute evaluation metrics.

    Args:
        model: Whisper model or PEFT Whisper model.
        processor: WhisperProcessor instance.
        dataset: Hugging Face dataset with 'audio' and 'text'.
        device: Torch device (CUDA/CPU).
        metrics: Metrics to evaluate (default: ['wer', 'cer']).
        logger: Logger instance.

    Returns:
        Tuple of (metric_results, detailed_sample_results).
    """
    import torch
    from tqdm.auto import tqdm
    from src.model import get_default_device

    if device is None:
        device = get_default_device()

    if logger is None:
        logger = setup_logger("evaluate")

    model.eval()
    predictions: List[str] = []
    references: List[str] = []
    sample_records: List[Dict[str, str]] = []

    logger.info(f"Starting inference on {len(dataset)} samples...")
    for i in tqdm(range(len(dataset)), desc="Evaluating"):
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
        pred_clean = transcription.strip().upper()
        ref_clean = ground_truth.strip().upper()

        predictions.append(pred_clean)
        references.append(ref_clean)
        sample_records.append({
            "index": i,
            "reference": ref_clean,
            "prediction": pred_clean,
        })

    metric_scores = compute_wer_cer(predictions, references, metrics=metrics)
    return metric_scores, sample_records


def load_model_for_eval(
    model_id: str = "openai/whisper-small",
    checkpoint_dir: Optional[str] = None,
    device: Optional[Any] = None,
) -> Tuple[Any, Any]:
    """Load base model or fine-tuned LoRA checkpoint for evaluation."""
    from peft import PeftModel
    from src.model import get_default_device, load_whisper_model_and_processor

    if device is None:
        device = get_default_device()

    base_model, processor = load_whisper_model_and_processor(model_id, device=device)

    if checkpoint_dir and os.path.exists(checkpoint_dir):
        # checkpoint- 로 시작하는 하위 디렉토리 목록 스캔
        subdirs = [
            d for d in os.listdir(checkpoint_dir)
            if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith("checkpoint-")
        ]

        if subdirs:
            # step 번호(숫자) 기준 또는 최종 수정 시간 기준으로 가장 최신 디렉토리 선택
            def _extract_step(d: str) -> int:
                parts = d.split("-")
                if len(parts) >= 2 and parts[-1].isdigit():
                    return int(parts[-1])
                return 0

            latest_subdir = max(
                subdirs,
                key=lambda d: (_extract_step(d), os.path.getmtime(os.path.join(checkpoint_dir, d))),
            )
            checkpoint_dir = os.path.join(checkpoint_dir, latest_subdir)
            print(f"[INFO] 발견된 최신 체크포인트 경로: {checkpoint_dir}")
        else:
            print(f"[INFO] 지정된 체크포인트 경로: {checkpoint_dir}")

        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    else:
        model = base_model

    return model.to(device), processor


def main() -> None:
    """CLI entry point for standalone evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Whisper on Tactical Speech Dataset")
    parser.add_argument("--model_id", type=str, default="openai/whisper-small")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to LoRA checkpoint")
    parser.add_argument("--dataset_name", type=str, default="openslr/librispeech_asr")
    parser.add_argument("--subset", type=str, default="clean")
    parser.add_argument("--split", type=str, default="validation[:50]")
    parser.add_argument("--degrade", action="store_true", default=True, help="Apply tactical degradation")
    parser.add_argument("--metrics", nargs="+", default=["wer", "cer"], help="Metrics to compute (wer, cer)")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save evaluation results JSON")
    args = parser.parse_args()

    logger = setup_logger("eval_cli")
    logger.info("Initializing evaluation pipeline...")

    from src.data_loader import load_speech_dataset
    from src.preprocess import TacticalAudioDegrader

    # 1. Load dataset
    dataset = load_speech_dataset(args.dataset_name, args.subset, split=args.split)

    # 2. Optionally apply tactical degradation
    if args.degrade:
        logger.info("Applying tactical audio degradation...")
        degrader = TacticalAudioDegrader()
        dataset = degrader.transform_dataset(dataset)

    # 3. Load model and processor
    model, processor = load_model_for_eval(args.model_id, args.checkpoint_dir)

    # 4. Evaluate
    metric_scores, sample_records = run_evaluation(
        model=model,
        processor=processor,
        dataset=dataset,
        metrics=args.metrics,
        logger=logger,
    )

    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    for metric_name, score in metric_scores.items():
        logger.info(f"  {metric_name.upper()}: {score * 100:.2f}%")
    logger.info("=" * 50)

    # 5. Save results if requested
    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        report = {
            "metrics": metric_scores,
            "config": vars(args),
            "samples": sample_records[:10],
        }
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved evaluation report to: {args.output_file}")


if __name__ == "__main__":
    main()

