"""
Utility functions and logging for the cognitive agent.
"""

import logging
import sys
from datetime import datetime
from typing import Any, Dict
import torch

# Setup logging
logger = logging.getLogger("CognitiveAgent")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def log_info(message: str) -> None:
    """Log informational message."""
    logger.info(message)


def log_debug(message: str) -> None:
    """Log debug message."""
    logger.debug(message)


def log_warning(message: str) -> None:
    """Log warning message."""
    logger.warning(message)


def log_error(message: str) -> None:
    """Log error message."""
    logger.error(message)


def tensor_to_dict(tensor: torch.Tensor) -> Dict[str, Any]:
    """Convert tensor to serializable dictionary."""
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std().item()),
    }


def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
