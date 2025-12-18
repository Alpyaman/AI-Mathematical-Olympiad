"""
Training Infrastructure for Mathematical Reasoning Model

This module provides the training infrastructure for Phase 2: Base Pre-training and beyond, including distributed training, mixed precision, and monitoring.
"""

from .pretrainer import PreTrainer
from .finetuner import FineTuner
from .distributed import setup_distributed, cleanup_distributed, is_distributed
from .config import PreTrainingConfig
from .finetuning_config import FineTuningConfig
from .robust_utils import (
        RobustDataCollator,
        fixed_train_step,
        validate_batch,
        safe_loss_computation,
        diagnose_batch_issue,
    )

 

__all__ = [
    "PreTrainer",
    "PreTrainingConfig",
    "setup_distributed",
    "cleanup_distributed",
    "is_distributed",
    "RobustDataCollator",
    "fixed_train_step",
    "validate_batch",
    "safe_loss_computation",
    "diagnose_batch_issue",
    "FineTuner",
    "FineTuningConfig",
]