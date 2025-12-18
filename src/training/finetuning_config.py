"""
Fine-Tuning Configuration

Configuration for Phase 2.2: Supervised Fine-Tuning on AIMO problems.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class FineTuningConfig:
    """Configuration for supervised fine-tuning."""

    # Model configuration
    model_config_name: str = "small"  # small, base, or large
    pretrained_checkpoint: Optional[str] = None  # Path to pre-trained checkpoint
    vocab_size: int = 50304
    max_seq_length: int = 2048

    # Data configuration
    train_data_path: str = "./data/reference.csv"
    val_data_path: Optional[str] = None  # Uses split from train if None
    test_data_path: Optional[str] = None
    train_ratio: float = 0.8
    val_ratio: float = 0.15
    test_ratio: float = 0.05

    # Training hyperparameters
    batch_size: int = 8
    micro_batch_size: int = 2  # Per-device batch size
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_steps: Optional[int] = None  # If set, overrides num_epochs
    warmup_steps: int = 100
    learning_rate: float = 2e-5  # Lower than pre-training
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Optimization
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    lr_scheduler: str = "cosine"  # cosine, linear, constant

    # Mixed precision
    mixed_precision: str = "bf16"  # fp16, bf16, or fp32
    gradient_checkpointing: bool = True

    # Distributed training
    distributed_backend: str = "nccl"  # nccl for GPU, gloo for CPU
    find_unused_parameters: bool = False

    # Checkpointing
    checkpoint_dir: str = "./checkpoints/finetuning"
    save_interval: int = 500  # Save every N steps
    keep_last_n_checkpoints: int = 3
    resume_from_checkpoint: Optional[str] = None

    # Evaluation
    eval_interval: int = 100  # Evaluate every N steps
    eval_steps: int = 50  # Number of eval steps per evaluation
    generate_samples: bool = True  # Generate sample solutions during eval
    num_eval_samples: int = 3  # Number of samples to generate

    # Logging
    log_interval: int = 10
    use_wandb: bool = False
    wandb_project: str = "math-reasoning-finetuning"
    wandb_run_name: Optional[str] = None
    use_tensorboard: bool = True
    tensorboard_dir: str = "./runs/finetuning"

    # System
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42

    # Fine-tuning specific
    freeze_embeddings: bool = False  # Freeze embedding layer
    freeze_layers: int = 0  # Number of initial layers to freeze

    def __post_init__(self):
        """Validate and compute derived values."""
        # Compute effective batch size
        self.effective_batch_size = (
            self.micro_batch_size *
            self.gradient_accumulation_steps
        )

        # Create directories
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        if self.use_tensorboard:
            Path(self.tensorboard_dir).mkdir(parents=True, exist_ok=True)

        # Validate learning rate
        if self.learning_rate >= 1e-3:
            print(f"⚠️  Warning: Fine-tuning LR ({self.learning_rate}) is high. "
                  f"Recommended: 1e-5 to 5e-5")

    def get_effective_batch_size(self, world_size: int = 1) -> int:
        """Get the effective global batch size."""
        return self.effective_batch_size * world_size

    def to_dict(self):
        """Convert to dictionary for logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}