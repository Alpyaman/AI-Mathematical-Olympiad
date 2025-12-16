"""
Baseline Training Script for Mathematical Reasoning Model

This script provides a simple but complete training setup to verify the entire pipeline works end-to-end.

Features:
- Training loop with gradient accumulation
- Loss tracking and logging
- Model checkpointing
- Validation evaluation
- Progress reporting
"""

import time
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Please install torch to run training.")
    exit(1)

from src import (
    get_small_config,
    MathTransformerDecoder,
    MathTokenizer,
)
from src.data import (
    create_sample_problems,
    DataPreprocessor,
    MathReasoningDataset,
    create_dataloaders,
    split_dataset,
)

class Trainer:
    """Simple trainer for mathematical reasoning model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 10,
    ):
        """
        Initialize trainer.

        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_interval: How often to log (in steps)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Statistics
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs["loss"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Log progress
            if self.global_step % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                print(f"  Step {self.global_step:4d} | Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | LR: {lr:.2e}")

        avg_epoch_loss = total_loss / num_batches
        return avg_epoch_loss

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in self.val_loader:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs["loss"]
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def save_checkpoint(self, filename: str = "checkpoint.pt"):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint to {path}")

    def load_checkpoint(self, filename: str = "checkpoint.pt"):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        if not path.exists():
            print(f"  Checkpoint {path} not found")
            return

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])

        print(f"  Loaded checkpoint from {path}")
        print(f"  Resuming from epoch {self.epoch}, step {self.global_step}")

    def train(self, num_epochs: int, save_every: int = 1):
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print()

        start_time = time.time()

        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_start = time.time()

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 60)

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            epoch_time = time.time() - epoch_start
 
            print(f"\n  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Time:       {epoch_time:.1f}s")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")
                print(f"  New best model! (Val Loss: {val_loss:.4f})")

            print()

        total_time = time.time() - start_time
        print(f"Training complete! Total time: {total_time / 60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

def create_optimizer(model, learning_rate: float = 5e-4, weight_decay: float = 0.01):
    """Create AdamW optimizer with weight decay."""
    # Separate parameters for weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Don't apply weight decay to biases and layer norms
        if "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer

def main():
    """Run baseline training."""
    print("\n" + "="*70)
    print("BASELINE TRAINING - End-to-End Pipeline Verification")
    print("="*70)

    # 1. Create sample data
    print("\n1. Creating sample dataset...")
    problems = create_sample_problems()
    print(f"   Created {len(problems)} sample problems")

    # Preprocess
    preprocessor = DataPreprocessor()
    processed = preprocessor.preprocess(problems, verbose=False)
    print(f"   Preprocessed: {len(processed)} problems ready")

    # Split data (ensure at least 1 validation sample)
    if len(processed) < 4:
        # For very small datasets, duplicate samples to ensure train and val both have data
        print(f"   Warning: Only {len(processed)} problems. Duplicating for train/val split...")
        train_problems = processed.copy()
        val_problems = processed.copy()  # Use same data for validation (just for demo)
        test_problems = []
    else:
        train_problems, val_problems, test_problems = split_dataset(
            processed,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
    )
    print(f"   Split: {len(train_problems)} train, {len(val_problems)} val, {len(test_problems)} test")

    # 2. Create datasets
    print("\n2. Creating PyTorch datasets...")
    tokenizer = MathTokenizer()

    train_dataset = MathReasoningDataset(
        problems=train_problems,
        tokenizer=tokenizer,
        max_length=512,
    )

    val_dataset = MathReasoningDataset(
        problems=val_problems,
        tokenizer=tokenizer,
        max_length=512,
    )

    # Create dataloaders
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from src.data.dataset import DataCollator
    from torch.utils.data import DataLoader

    collator = DataCollator(pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
        pin_memory=(device == "cuda"),  # Only pin memory if using GPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        pin_memory=(device == "cuda"),
    )

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    # 3. Create model
    print("\n3. Initializing model...")
    config = get_small_config()
    # Make even smaller for quick training
    config.hidden_size = 256
    config.num_hidden_layers = 4
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.intermediate_size = 1024
    config.max_position_embeddings = 512
    config.max_sequence_length = 512

    model = MathTransformerDecoder(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model: {total_params:,} parameters (~{total_params * 4 / (1024**2):.1f}MB)")
    print(f"   Config: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")

    # 4. Create optimizer and scheduler
    print("\n4. Setting up training...")
    num_epochs = 10
    learning_rate = 1e-3
    num_training_steps = len(train_loader) * num_epochs

    optimizer = create_optimizer(model, learning_rate=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-5)

    print(f"   Optimizer: AdamW (lr={learning_rate})")
    print("   Scheduler: CosineAnnealingLR")
    print(f"   Epochs: {num_epochs}")
    print(f"   Total steps: {num_training_steps}")

    # 5. Train
    print("\n5. Training model...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir="./checkpoints/baseline",
        log_interval=5,
    )

    trainer.train(num_epochs=num_epochs, save_every=5)

    # 6. Summary
    print("\n" + "="*70)
    print("BASELINE TRAINING COMPLETE ✓")
    print("="*70)
    print("\nTraining Summary:")
    print(f"  Total epochs: {num_epochs}")
    print(f"  Total steps: {trainer.global_step}")
    print(f"  Best val loss: {trainer.best_val_loss:.4f}")
    print(f"  Final train loss: {trainer.train_losses[-1]:.4f}")
    print(f"  Final val loss: {trainer.val_losses[-1]:.4f}")

    print(f"\nCheckpoints saved to: {trainer.checkpoint_dir}")
    print("\nNext steps:")
    print("  1. Train on larger dataset (MATH, AoPS)")
    print("  2. Implement evaluation metrics")
    print("  3. Add distributed training support")
    print("  4. Proceed to Phase 2: Full Training Infrastructure")

if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("Error: PyTorch is required for training")
        print("Install with: pip install torch")
        exit(1)
 
    main()