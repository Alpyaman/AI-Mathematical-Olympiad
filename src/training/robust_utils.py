"""
Robust Data Utilities for Pre-Training

This module provides robust data handling components that gracefully handle edge cases like variable lengths, empty sequences, and dimension mismatches.
"""

import torch
from typing import List, Dict, Any, Optional
import warnings

class RobustDataCollator:
    """
    Robust data collator that handles edge cases in pre-training.

    Improvements over basic collator:
    - Handles sequences of different lengths safely
    - Validates tensor shapes before batching
    - Clips extremely long sequences
    - Handles empty or invalid examples gracefully
    - Better error messages for debugging
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        max_length: Optional[int] = None,
        return_tensors: bool = True,
    ):
        """
        Initialize robust collator.

        Args:
            pad_token_id: Token ID for padding
            max_length: Maximum sequence length (clips longer sequences)
            return_tensors: Whether to return PyTorch tensors
        """
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.return_tensors = return_tensors

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch with robust error handling.

        Args:
            batch: List of examples from dataset

        Returns:
            Batched tensors with proper padding and masking
        """
        if not batch:
            raise ValueError("Cannot collate empty batch")

        # Filter out None or invalid examples
        valid_batch = []
        for i, example in enumerate(batch):
            if example is None:
                warnings.warn(f"Skipping None example at index {i}")
                continue

            if "input_ids" not in example:
                warnings.warn(f"Skipping example without input_ids at index {i}")
                continue

            # Check if input_ids is valid
            input_ids = example["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                if input_ids.numel() == 0:
                    warnings.warn(f"Skipping empty input_ids at index {i}")
                    continue
            elif isinstance(input_ids, list):
                if len(input_ids) == 0:
                    warnings.warn(f"Skipping empty input_ids at index {i}")
                    continue

            valid_batch.append(example)

        if not valid_batch:
            raise ValueError("All examples in batch were invalid")

        # Convert to tensors if needed
        for example in valid_batch:
            if not isinstance(example["input_ids"], torch.Tensor):
                example["input_ids"] = torch.tensor(example["input_ids"], dtype=torch.long)
            if "attention_mask" in example and not isinstance(example["attention_mask"], torch.Tensor):
                example["attention_mask"] = torch.tensor(example["attention_mask"], dtype=torch.long)

        # Find max length in batch
        max_len = max(example["input_ids"].shape[0] for example in valid_batch)

        # Clip to max_length if specified
        if self.max_length is not None and max_len > self.max_length:
            max_len = self.max_length
            for example in valid_batch:
                if example["input_ids"].shape[0] > max_len:
                    example["input_ids"] = example["input_ids"][:max_len]
                    if "attention_mask" in example:
                        example["attention_mask"] = example["attention_mask"][:max_len]

        # Prepare batched tensors
        batch_size = len(valid_batch)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        # Fill in the tensors
        for i, example in enumerate(valid_batch):
            seq_len = example["input_ids"].shape[0]

            # Ensure seq_len doesn't exceed max_len
            seq_len = min(seq_len, max_len)

            # Copy input_ids
            input_ids[i, :seq_len] = example["input_ids"][:seq_len]

            # Copy or create attention_mask
            if "attention_mask" in example:
                attention_mask[i, :seq_len] = example["attention_mask"][:seq_len]
            else:
                attention_mask[i, :seq_len] = 1

            # Create labels for causal LM
            # Labels are input_ids shifted by 1 (predict next token)
            if "labels" in example:
                labels[i, :seq_len] = example["labels"][:seq_len]
            else:
                labels[i, :seq_len] = example["input_ids"][:seq_len]
                # Mask padding in labels
                labels[i, :seq_len][attention_mask[i, :seq_len] == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def validate_batch(batch: Dict[str, torch.Tensor]) -> bool:
    """
    Validate a batch before training.

    Args:
        batch: Batch dictionary

    Returns:
        True if valid, False otherwise
    """
    required_keys = ["input_ids", "attention_mask", "labels"]

    for key in required_keys:
        if key not in batch:
            warnings.warn(f"Missing required key: {key}")
            return False

    # Check shapes match
    batch_size = batch["input_ids"].shape[0]
    seq_len = batch["input_ids"].shape[1]

    if batch["attention_mask"].shape != (batch_size, seq_len):
        warnings.warn(
            f"Shape mismatch: input_ids {batch['input_ids'].shape} "
            f"vs attention_mask {batch['attention_mask'].shape}"
        )
        return False

    if batch["labels"].shape != (batch_size, seq_len):
        warnings.warn(
            f"Shape mismatch: input_ids {batch['input_ids'].shape} "
            f"vs labels {batch['labels'].shape}"
        )
        return False

    return True

def safe_loss_computation(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Safely compute cross-entropy loss with proper shape handling.

    Args:
        logits: Model output logits (batch_size, seq_len, vocab_size)
        labels: Target labels (batch_size, seq_len)

    Returns:
        Scalar loss
    """
    # Validate shapes
    if logits.dim() != 3:
        raise ValueError(f"Expected logits to be 3D (batch, seq, vocab), got shape {logits.shape}")
 
    if labels.dim() != 2:
        raise ValueError(f"Expected labels to be 2D (batch, seq), got shape {labels.shape}")

    batch_size, seq_len, vocab_size = logits.shape

    if labels.shape[0] != batch_size or labels.shape[1] != seq_len:
        raise ValueError(
            f"Shape mismatch: logits {logits.shape} vs labels {labels.shape}"
        )

    # Reshape for loss computation
    # logits: (batch_size * seq_len, vocab_size)
    # labels: (batch_size * seq_len,)
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)

    # Compute loss (ignore_index=-100 handles padding)
    loss = torch.nn.functional.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=-100,
        reduction='mean'
    )

    return loss

def fixed_train_step(
    model,
    batch: Dict[str, torch.Tensor],
    device: str,
    use_amp: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> float:
    """
    Robust training step with proper error handling.

    Args:
        model: Model to train
        batch: Training batch
        device: Device to use
        use_amp: Whether to use automatic mixed precision
        dtype: Data type for mixed precision

    Returns:
        Loss value
    """
    # Validate batch
    if not validate_batch(batch):
        raise ValueError("Invalid batch structure")

    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}

    # Forward pass with mixed precision
    if use_amp and dtype is not None:
        with torch.amp.autocast("cuda", dtype=dtype):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            # Handle different output formats
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            # Compute loss safely
            loss = safe_loss_computation(logits, batch["labels"])
    else:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        # Handle different output formats
        if isinstance(outputs, tuple):
            logits = outputs[0]
        elif isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs

        # Compute loss safely
        loss = safe_loss_computation(logits, batch["labels"])

    return loss

def diagnose_batch_issue(batch: Dict[str, torch.Tensor]) -> str:
    """
    Diagnose common issues with a batch.

    Args:
        batch: Batch to diagnose

    Returns:
        Diagnostic message
    """
    messages = []

    if "input_ids" in batch:
        messages.append(f"input_ids shape: {batch['input_ids'].shape}")
        messages.append(f"input_ids dtype: {batch['input_ids'].dtype}")
        messages.append(f"input_ids min/max: {batch['input_ids'].min()}/{batch['input_ids'].max()}")
    else:
        messages.append("❌ Missing input_ids")

    if "attention_mask" in batch:
        messages.append(f"attention_mask shape: {batch['attention_mask'].shape}")
        messages.append(f"attention_mask sum: {batch['attention_mask'].sum()}")
    else:
        messages.append("❌ Missing attention_mask")

    if "labels" in batch:
        messages.append(f"labels shape: {batch['labels'].shape}")
        valid_labels = (batch['labels'] != -100).sum()
        messages.append(f"valid labels: {valid_labels}")
    else:
        messages.append("❌ Missing labels")

    return "\n".join(messages)