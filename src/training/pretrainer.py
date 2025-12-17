"""
Pre-Trainer for Phase 2.1: Base Pre-training

Advanced training infrastructure with distributed training, mixed precision,
gradient accumulation, and comprehensive monitoring.
"""

import time
import math
from pathlib import Path
from typing import Optional, Dict, Any

from .config import PreTrainingConfig

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    from torch.cuda.amp import autocast, GradScaler
    from torch.nn.parallel import DistributedDataParallel as DDP

    from .distributed import (
        setup_distributed,
        cleanup_distributed,
        is_main_process,
        is_distributed,
        barrier,
        reduce_dict,
        save_on_main_process,
    )

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise ImportError(
        "PyTorch is required for the PreTrainer class. "
        "Please install torch to use pre-training functionality: "
        "pip install torch"
    )


class PreTrainer:
    """
    Pre-training infrastructure for Phase 2.1: Base Pre-training.

    Features:
    - Distributed training (DDP)
    - Mixed precision (fp16/bf16)
    - Gradient accumulation
    - Gradient checkpointing
    - Learning rate scheduling
    - Checkpointing with auto-resume
    - Logging and monitoring
    """

    def __init__(
        self,
        model: "nn.Module",
        config: PreTrainingConfig,
        train_dataloader,
        val_dataloader: Optional[Any] = None,
    ):
        """
        Initialize pre-trainer.

        Args:
            model: Model to train
            config: Pre-training configuration
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training")

        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Setup distributed training
        self.dist_info = setup_distributed(backend=config.distributed_backend)
        self.device = self.dist_info["device"]
        self.is_main_process = self.dist_info["is_main_process"]

        # Move model to device
        self.model = model.to(self.device)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()

        # Wrap model with DDP if distributed
        if self.dist_info["is_distributed"]:
            self.model = DDP(
                self.model,
                device_ids=[self.dist_info["local_rank"]],
                output_device=self.dist_info["local_rank"],
                find_unused_parameters=config.find_unused_parameters,
            )
            self.raw_model = self.model.module
        else:
            self.raw_model = self.model

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Setup mixed precision
        self.use_amp = config.mixed_precision in ["fp16", "bf16"]
        self.dtype = self._get_dtype()
        self.scaler = GradScaler() if config.mixed_precision == "fp16" else None

        # Setup logging
        self.logger = self._setup_logging()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.tokens_seen = 0

        # Load checkpoint if resuming
        if config.resume_from_checkpoint:
            self.load_checkpoint(config.resume_from_checkpoint)

        # Log configuration
        if self.is_main_process:
            self._log_config()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay."""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.raw_model.named_parameters():
            if not param.requires_grad:
                continue

            # No weight decay for biases and layer norms
            if any(nd in name for nd in ["bias", "norm", "ln_f"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

        return optimizer

    def _create_scheduler(self) -> LambdaLR:
        """Create learning rate scheduler."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                # Linear warmup
                return step / max(1, self.config.warmup_steps)
            else:
                # Cosine decay
                progress = (step - self.config.warmup_steps) / max(
                    1, self.config.max_steps - self.config.warmup_steps
                )
                return max(
                    self.config.min_learning_rate / self.config.learning_rate,
                    0.5 * (1.0 + math.cos(math.pi * progress))
                )

        return LambdaLR(self.optimizer, lr_lambda)

    def _get_dtype(self):
        """Get dtype for mixed precision."""
        if self.config.mixed_precision == "bf16":
            return torch.bfloat16
        elif self.config.mixed_precision == "fp16":
            return torch.float16
        else:
            return torch.float32

    def _setup_logging(self):
        """Setup logging (wandb/tensorboard)."""
        logger = {}

        if self.is_main_process:
            # Setup TensorBoard
            if self.config.use_tensorboard:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    logger['tensorboard'] = SummaryWriter(self.config.tensorboard_dir)
                except ImportError:
                    print("TensorBoard not available. Install tensorboard to use logging.")

            # Setup Weights & Biases
            if self.config.use_wandb:
                try:
                    import wandb
                    wandb.init(
                        project=self.config.wandb_project,
                        name=self.config.wandb_run_name,
                        config=self.config.to_dict(),
                    )
                    logger['wandb'] = wandb
                except ImportError:
                    print("Wandb not available. Install wandb to use logging.")

        return logger

    def _log_config(self):
        """Log configuration."""
        print("\n" + "=" * 70)
        print("PRE-TRAINING CONFIGURATION")
        print("=" * 70)
        print(f"Model: {self.config.model_config_name}")
        print(f"Max steps: {self.config.max_steps:,}")
        print(f"Batch size: {self.config.micro_batch_size} (per device)")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps} steps")
        print(f"Effective batch size: {self.config.get_effective_batch_size(self.dist_info['world_size']):,}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Mixed precision: {self.config.mixed_precision}")
        print(f"Gradient checkpointing: {self.config.gradient_checkpointing}")
        print(f"World size: {self.dist_info['world_size']}")
        print(f"Device: {self.device}")
        print("=" * 70 + "\n")

    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to all configured loggers."""
        if not self.is_main_process:
            return

        # Log to TensorBoard
        if 'tensorboard' in self.logger:
            for key, value in metrics.items():
                self.logger['tensorboard'].add_scalar(key, value, step)

        # Log to Wandb
        if 'wandb' in self.logger:
            self.logger['wandb'].log(metrics, step=step)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Perform a single training step.

        Args:
            batch: Batch of data

        Returns:
            Loss value
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass with mixed precision
        with autocast(dtype=self.dtype, enabled=self.use_amp):
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            logits = outputs  # Model returns logits directly

            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch["labels"].reshape(-1),
                ignore_index=-100,
            )

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps

    def train(self):
        """Main training loop."""
        print(f"\nStarting pre-training from step {self.global_step}...")
        start_time = time.time()

        self.model.train()
        accumulated_loss = 0.0
        step_times = []

        for batch in self.train_dataloader:
            step_start = time.time()

            # Training step
            loss = self.train_step(batch)
            accumulated_loss += loss

            # Gradient accumulation
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                if self.config.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                # Update tokens seen
                batch_size = batch["input_ids"].size(0)
                seq_length = batch["input_ids"].size(1)
                self.tokens_seen += batch_size * seq_length * self.dist_info["world_size"]

                # Average loss
                avg_loss = accumulated_loss / self.config.gradient_accumulation_steps
                accumulated_loss = 0.0

                # Logging
                if self.global_step % self.config.log_interval == 0:
                    step_time = time.time() - step_start
                    step_times.append(step_time)

                    # Compute metrics
                    metrics = {
                        "train/loss": avg_loss,
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/tokens_seen": self.tokens_seen,
                        "train/step": self.global_step,
                        "train/step_time": step_time,
                    }

                    if len(step_times) > 0:
                        metrics["train/tokens_per_second"] = (
                            batch_size * seq_length * self.dist_info["world_size"] /
                            (sum(step_times) / len(step_times))
                        )

                    # Reduce metrics across processes
                    if is_distributed():
                        loss_tensor = torch.tensor(avg_loss, device=self.device)
                        metrics_reduced = reduce_dict({"loss": loss_tensor})
                        metrics["train/loss"] = metrics_reduced["loss"]

                    # Log metrics
                    self._log_metrics(metrics, self.global_step)

                    if self.is_main_process:
                        elapsed = time.time() - start_time
                        print(
                            f"Step {self.global_step}/{self.config.max_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {metrics['train/learning_rate']:.2e} | "
                            f"Tokens: {self.tokens_seen:,} | "
                            f"Time: {elapsed:.1f}s"
                        )

                # Validation
                if (self.val_dataloader is not None and
                    self.global_step % self.config.eval_interval == 0 and
                    self.global_step > 0):
                    val_loss = self.validate()

                    if self.is_main_process:
                        print(f"Validation loss: {val_loss:.4f}")

                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        if self.is_main_process:
                            self.save_checkpoint("best.pt")

                # Checkpointing
                if (self.global_step % self.config.save_interval == 0 and
                    self.global_step > 0):
                    if self.is_main_process:
                        self.save_checkpoint(f"step_{self.global_step}.pt")
                        self._cleanup_old_checkpoints()

            self.global_step += 1

            # Check if we've reached max steps
            if self.global_step >= self.config.max_steps:
                break

        # Final checkpoint
        if self.is_main_process:
            self.save_checkpoint("final.pt")

        total_time = time.time() - start_time
        if self.is_main_process:
            print(f"\nPre-training complete! Total time: {total_time / 3600:.2f} hours")
            print(f"Total tokens processed: {self.tokens_seen:,}")

        # Cleanup
        cleanup_distributed()

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_dataloader:
            if num_batches >= self.config.eval_steps:
                break

            batch = {k: v.to(self.device) for k, v in batch.items()}

            with autocast(dtype=self.dtype, enabled=self.use_amp):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                logits = outputs

                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch["labels"].reshape(-1),
                    ignore_index=-100,
                )

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)

        # Reduce across processes
        if is_distributed():
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            metrics = reduce_dict({"loss": loss_tensor})
            avg_loss = metrics["loss"]

        # Log validation metrics
        if self.is_main_process:
            self._log_metrics({"val/loss": avg_loss}, self.global_step)

        self.model.train()
        return avg_loss

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "tokens_seen": self.tokens_seen,
            "config": self.config.to_dict(),
        }

        if self.scaler is not None:
            checkpoint["scaler"] = self.scaler.state_dict()

        checkpoint_path = Path(self.config.checkpoint_dir) / filename
        save_on_main_process(checkpoint, str(checkpoint_path))

        if self.is_main_process:
            print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}...")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.raw_model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        self.tokens_seen = checkpoint.get("tokens_seen", 0)

        if self.scaler is not None and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        print(f"Resumed from step {self.global_step}")

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent N."""
        if not self.is_main_process:
            return

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoints = sorted(
            [f for f in checkpoint_dir.glob("step_*.pt")],
            key=lambda x: int(x.stem.split("_")[1])
        )

        # Keep only the last N checkpoints
        if len(checkpoints) > self.config.keep_last_n_checkpoints:
            for checkpoint in checkpoints[:-self.config.keep_last_n_checkpoints]:
                checkpoint.unlink()
                print(f"Removed old checkpoint: {checkpoint}")