"""
PyTorch Dataset for Mathematical Reasoning

This module provides PyTorch-compatible dataset classes for training
mathematical reasoning models.
"""

from typing import List, Optional, Dict, Any
import random

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy base class when torch is not available
    class Dataset:
        pass

from .data_schema import MathProblem
from .data_formatter import ChainOfThoughtFormatter
from ..tokenizer.math_tokenizer import MathTokenizer


class MathReasoningDataset(Dataset):
    """
    PyTorch Dataset for mathematical reasoning problems.

    This dataset handles tokenization, formatting, and batching of
    mathematical problems for training.
    """

    def __init__(
        self,
        problems: List[MathProblem],
        tokenizer: MathTokenizer,
        formatter: Optional[ChainOfThoughtFormatter] = None,
        max_length: int = 2048,
        include_solution: bool = True,
        add_eos_token: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            problems: List of mathematical problems
            tokenizer: Tokenizer for encoding text
            formatter: CoT formatter (creates default if None)
            max_length: Maximum sequence length
            include_solution: Whether to include solutions in training
            add_eos_token: Whether to add EOS token
        """
        self.problems = problems
        self.tokenizer = tokenizer
        self.formatter = formatter or ChainOfThoughtFormatter(use_special_tokens=True)
        self.max_length = max_length
        self.include_solution = include_solution
        self.add_eos_token = add_eos_token

    def __len__(self) -> int:
        """Return the number of problems in the dataset."""
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

        # Format the problem
        text = self.formatter.format_problem(problem, include_solution=self.include_solution)

        # Tokenize
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=self.max_length,
            padding=True,
            truncation=True
        )

        if TORCH_AVAILABLE:
            # Convert to tensors
            input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)

            # For language modeling, labels are the same as input_ids
            # (model predicts next token)
            labels = input_ids.clone()

            # Optionally mask padding tokens in labels
            labels[attention_mask == 0] = -100  # -100 is ignored by CrossEntropyLoss
        else:
            # Return raw lists when torch not available
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            labels = [x if m == 1 else -100 for x, m in zip(input_ids, attention_mask)]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "problem_id": problem.problem_id,
        }

    def get_problem(self, idx: int) -> MathProblem:
        """Get the raw problem object at index."""
        return self.problems[idx]

    def filter_by_difficulty(self, difficulties: List[str]) -> "MathReasoningDataset":
        """
        Create a new dataset filtered by difficulty.

        Args:
            difficulties: List of difficulty levels to include

        Returns:
            New filtered dataset
        """
        filtered_problems = [
            p for p in self.problems
            if p.difficulty.value in difficulties
        ]

        return MathReasoningDataset(
            problems=filtered_problems,
            tokenizer=self.tokenizer,
            formatter=self.formatter,
            max_length=self.max_length,
            include_solution=self.include_solution,
            add_eos_token=self.add_eos_token,
        )

    def filter_by_type(self, problem_types: List[str]) -> "MathReasoningDataset":
        """
        Create a new dataset filtered by problem type.

        Args:
            problem_types: List of problem types to include

        Returns:
            New filtered dataset
        """
        filtered_problems = [
            p for p in self.problems
            if p.problem_type.value in problem_types
        ]

        return MathReasoningDataset(
            problems=filtered_problems,
            tokenizer=self.tokenizer,
            formatter=self.formatter,
            max_length=self.max_length,
            include_solution=self.include_solution,
            add_eos_token=self.add_eos_token,
        )


class DataCollator:
    """
    Custom data collator for batching mathematical reasoning examples.
    """

    def __init__(self, pad_token_id: int = 0):
        """
        Initialize the collator.

        Args:
            pad_token_id: Token ID to use for padding
        """
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of examples.

        Args:
            batch: List of examples from __getitem__

        Returns:
            Batched tensors
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for batching. Install torch to use DataCollator.")

        # Find max length in batch
        max_length = max(example["input_ids"].shape[0] for example in batch)

        # Prepare batched tensors
        batch_size = len(batch)
        input_ids = torch.full((batch_size, max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels = torch.full((batch_size, max_length), -100, dtype=torch.long)

        # Fill in the tensors
        for i, example in enumerate(batch):
            seq_len = example["input_ids"].shape[0]
            input_ids[i, :seq_len] = example["input_ids"]
            attention_mask[i, :seq_len] = example["attention_mask"]
            labels[i, :seq_len] = example["labels"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_dataloaders(
    train_dataset: MathReasoningDataset,
    val_dataset: Optional[MathReasoningDataset] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle_train: bool = True,
    collator: Optional[DataCollator] = None,
) -> tuple:
    """
    Create PyTorch DataLoaders for training and validation.

    Args:
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle_train: Whether to shuffle training data
        collator: Optional custom data collator

    Returns:
        Tuple of (train_loader, val_loader) or just train_loader
    """
    if collator is None:
        collator = DataCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
        )
        return train_loader, val_loader

    return train_loader


def split_dataset(
    problems: List[MathProblem],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple:
    """
    Split problems into train/val/test sets.

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


def stratified_split(
    problems: List[MathProblem],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    stratify_by: str = "difficulty",
    seed: int = 42,
) -> tuple:
    """
    Split problems with stratification by difficulty or type.

    Args:
        problems: List of problems to split
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        stratify_by: Attribute to stratify by ('difficulty' or 'type')
        seed: Random seed

    Returns:
        Tuple of (train_problems, val_problems, test_problems)
    """
    random.seed(seed)

    # Group problems by stratification attribute
    groups = {}
    for problem in problems:
        if stratify_by == "difficulty":
            key = problem.difficulty.value
        elif stratify_by == "type":
            key = problem.problem_type.value
        else:
            raise ValueError(f"Unknown stratify_by: {stratify_by}")

        if key not in groups:
            groups[key] = []
        groups[key].append(problem)

    # Split each group
    train_problems = []
    val_problems = []
    test_problems = []

    for key, group_problems in groups.items():
        random.shuffle(group_problems)

        n = len(group_problems)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_problems.extend(group_problems[:train_end])
        val_problems.extend(group_problems[train_end:val_end])
        test_problems.extend(group_problems[val_end:])

    # Shuffle the combined splits
    random.shuffle(train_problems)
    random.shuffle(val_problems)
    random.shuffle(test_problems)

    return train_problems, val_problems, test_problems