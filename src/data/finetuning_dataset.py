"""
PyTorch Dataset for Fine-Tuning on AIMO Problems

This module provides PyTorch datasets for supervised fine-tuning on
mathematical reasoning tasks.
"""

from typing import List, Optional, Dict, Any
import random

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class Dataset:
        pass

from .aimo_dataset import AIMOProblem, AIMOFormatter
from ..tokenizer.math_tokenizer import MathTokenizer


class AIMOFineTuningDataset(Dataset):
    """
    PyTorch Dataset for fine-tuning on AIMO problems.

    This dataset formats AIMO problems for supervised learning,
    where the model learns to generate step-by-step solutions
    and final answers.
    """

    def __init__(self, problems: List[AIMOProblem], tokenizer: MathTokenizer, formatter: Optional[AIMOFormatter] = None, max_length: int = 2048, add_eos_token: bool = True):
        """
        Initialize fine-tuning dataset.

        Args:
            problems: List of AIMO problems
            tokenizer: Tokenizer for encoding
            formatter: Problem formatter (creates default if None)
            max_length: Maximum sequence length
            add_eos_token: Whether to add EOS token
        """
        self.problems = problems
        self.tokenizer = tokenizer
        self.formatter = formatter or AIMOFormatter(
            use_special_tokens=True,
            include_reasoning=True,
        )
        self.max_length = max_length
        self.add_eos_token = add_eos_token

    def __len__(self) -> int:
        """Return number of problems."""
        return len(self.problems)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single training example.

        Args:
            idx: Index of the problem

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        problem = self.problems[idx]

        # Format problem with solution and answer
        text = self.formatter.format_for_training(problem)

        # Tokenize
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding=True,
            truncation=True
        )

        if TORCH_AVAILABLE:
            # Convert to tensors
            input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)

            # For fine-tuning, labels are the same as input_ids
            # Model learns to predict next token
            labels = input_ids.clone()

            # Mask padding tokens in labels
            labels[attention_mask == 0] = -100

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "problem_id": problem.id,
                "answer": problem.answer,
            }
        else:
            # Return raw lists when torch not available
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            labels = [x if m == 1 else -100 for x, m in zip(input_ids, attention_mask)]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "problem_id": problem.id,
                "answer": problem.answer,
            }

    def get_problem(self, idx: int) -> AIMOProblem:
        """Get the raw problem at index."""
        return self.problems[idx]


def split_aimo_dataset(problems: List[AIMOProblem], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, shuffle: bool = True, seed: int = 42) -> tuple:
    """
    Split AIMO problems into train/val/test sets.

    Args:
        problems: List of problems to split
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        shuffle: Whether to shuffle before splitting
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_problems, val_problems, test_problems)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    if shuffle:
        random.seed(seed)
        problems = problems.copy()
        random.shuffle(problems)

    n = len(problems)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_problems = problems[:train_end]
    val_problems = problems[train_end:val_end]
    test_problems = problems[val_end:]

    return train_problems, val_problems, test_problems