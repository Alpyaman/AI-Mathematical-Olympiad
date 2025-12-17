"""
Pre-training Configuration

Configuration for Phase 2.1: Base Pre-training
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class PreTrainingConfig:
    """Configuration for base pre-training."""

    # Model configuration
    model_config_name: str = "base" # Small, Base, or Large
    vocab_size: int = 50304
    max_seq_length: int = 2048

    # Data Configuration
    data_dir: Path = Path("./data/pretraining")
    data_sources: List[str] = field(default_factory=lambda: ["arxiv", "general"])
    data_mix_weights: List[float] = field(default_factory=lambda: [0.3, 0.7]) # Arxiv 30%, General 70%
    streaming: bool = True # Use streaming for large datasets
    preprocessing_workers: int = 4

    # Training Hyperparameters
    batch_size: int = 32
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_steps: int = 1000000
    warmup_steps: int = 2000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Optimization
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    lr_scheduler: str = "cosine" # Options: linear, cosine, constant

    # Mixed Precision
    mixed_precision: str = "bf16" # Options: fp16, bf16, or fp32
    gradient_checkpointing: bool = True

    # Distributed Training
    distributed_backend: str = "nccl" # nccl for GPUs, gloo for CPUs
    find_unused_parameters: bool = False

    # Checkpointing
    checkpoint_dir: Path = Path("./checkpoints/pretraining")
    save_interval: int = 5000  # Save every N steps
    keep_last_n_checkpoints: int = 3
    resume_from_checkpoint: Optional[str] = None

    # Logging
    log_interval: int = 10
    eval_interval: int = 1000
    eval_steps: int = 100
    use_wandb: bool = False
    wandb_project: str = "math-reasoning-pretraining"
    wandb_run_name: Optional[str] = None
    use_tensorboard: bool = True
    tensorboard_dir: str = "./runs/pretraining"

    # Validation
    val_data_dir: Optional[str] = None
    val_batch_size: int = 32

    # System
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42

    def __post_init__(self):
        """Validate and compute derived values."""
        # Compute effective batch size
        self.effective_batch_size = (
            self.micro_batch_size *
            self.gradient_accumulation_steps
        )

        # Validate data mix weights
        if len(self.data_sources) != len(self.data_mix_weights):
            raise ValueError(
                f"Number of data sources ({len(self.data_sources)}) must match "
                f"number of mix weights ({len(self.data_mix_weights)})"
            )

        # Normalize weights
        total_weight = sum(self.data_mix_weights)
        self.data_mix_weights = [w / total_weight for w in self.data_mix_weights]

        # Create directories
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        if self.use_tensorboard:
            Path(self.tensorboard_dir).mkdir(parents=True, exist_ok=True)

    def get_effective_batch_size(self, world_size: int = 1) -> int:
        """Get the effective global batch size."""
        return self.effective_batch_size * world_size

    def to_dict(self):
        """Convert to dictionary for logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}