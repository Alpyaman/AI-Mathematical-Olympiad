"""
Training Infrastructure for Mathematical Reasoning Model

This module provides the training infrastructure for Phase 2: Base Pre-training and beyond, including distributed training, mixed precision, and monitoring.
"""

from .pretrainer import Pretrainer
from .distributed import setup_distributed, cleanup_distributed, is_distributed
from .config import PreTrainingConfig

__all__ = [
    "Pretrainer",
    "setup_distributed",
    "cleanup_distributed",
    "is_distributed",
    "PreTrainingConfig",
]