"""Utility module for tactical speech recognition.

Provides random seed fixation for experiment reproducibility,
standardized logging configuration, and YAML configuration loading.
"""

import logging
import os
import random
import sys
from typing import Any, Dict, Optional
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across python, numpy, torch for reproducible experiments.

    Args:
        seed: Random seed integer value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except ImportError:
        pass

    print(f"[Seed] Random seed fixed to: {seed}")


def setup_logger(
    name: str = "tactical_speech",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a standardized logger with console and optional file handler.

    Args:
        name: Name of the logger instance.
        log_file: Optional path to output log file.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if function is called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Safely load and parse a YAML configuration file.

    Args:
        config_path: Path to .yaml file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    import yaml

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}

