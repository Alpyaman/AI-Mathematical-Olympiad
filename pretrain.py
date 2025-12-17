"""
Phase 2.1: Base Pre-Training Script

This script implements base pre-training with causal language modeling on a mixture of mathematical (ArXiv) and general text corpora.

Usage:
    # Single GPU training
    python pretrain.py

    # Multi-GPU training (DDP)
    torchrun --nproc_per_node=4 pretrain.py

    # Resume from checkpoint
    python pretrain.py --resume checkpoints/pretraining/step_10000.pt

    # Custom configuration
    python pretrain.py --config configs/pretrain_large.json
"""

import argparse
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Please install torch to run pre-training.")
    exit(1)

from src import (
    get_small_config,
    get_base_config,
    get_large_config,
    MathTransformerDecoder,
    MathTokenizer,
)
from src.training import PreTrainer, PreTrainingConfig
from src.data.pretraining_dataset import (
    prepare_pretraining_data,
    PreTrainingDataCollator,
    create_sample_pretraining_data,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Phase 2.1: Base Pre-Training")

    # Model
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["small", "base", "large"],
        help="Model size preset"
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/pretraining",
        help="Directory containing pre-training data"
    )
    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="Create sample data for testing"
    )

    # Training
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Micro batch size per device"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100000,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Peak learning rate"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2000,
        help="Warmup steps"
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Mixed precision training"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/pretraining",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5000,
        help="Save checkpoint every N steps"
    )

    # Logging
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log every N steps"
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="math-reasoning-pretraining",
        help="Wandb project name"
    )

    # System
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    return parser.parse_args()


def get_model_config(size: str):
    """Get model configuration by size."""
    if size == "small":
        return get_small_config()
    elif size == "base":
        return get_base_config()
    elif size == "large":
        return get_large_config()
    else:
        raise ValueError(f"Unknown model size: {size}")


def main():
    """Main pre-training function."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    print("\n" + "=" * 70)
    print("PHASE 2.1: BASE PRE-TRAINING")
    print("=" * 70)

    # Create sample data if requested
    if args.create_sample_data:
        print("\nCreating sample pre-training data...")
        create_sample_pretraining_data(args.data_dir)
        print("Sample data created successfully!")

    # Verify data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"\nError: Data directory not found: {data_dir}")
        print("Please provide pre-training data or use --create-sample-data")
        return

    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = MathTokenizer()

    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = prepare_pretraining_data(
        data_dir=args.data_dir,
        sources=["arxiv", "general"],
        tokenizer=tokenizer,
        max_seq_length=2048,
        mix_weights=[0.3, 0.7],  # 30% math, 70% general
    )

    # Create data loader
    collator = PreTrainingDataCollator(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    print("Training data prepared with mixed domain sampling")
    print("  - ArXiv (mathematical): 30%")
    print("  - General text: 70%")

    # Initialize model
    print(f"\nInitializing {args.model_size} model...")
    model_config = get_model_config(args.model_size)

    # Update vocab size to match tokenizer
    model_config.vocab_size = len(tokenizer)

    model = MathTransformerDecoder(model_config)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_trainable:,} trainable)")

    # Create training configuration
    training_config = PreTrainingConfig(
        model_config_name=args.model_size,
        vocab_size=model_config.vocab_size,
        max_seq_length=model_config.max_position_embeddings,
        data_dir=args.data_dir,
        micro_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        mixed_precision=args.mixed_precision,
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.save_interval,
        resume_from_checkpoint=args.resume,
        log_interval=args.log_interval,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = PreTrainer(
        model=model,
        config=training_config,
        train_dataloader=train_loader,
        val_dataloader=None,  # Add validation later if needed
    )

    # Start training
    print("\n" + "=" * 70)
    print("STARTING BASE PRE-TRAINING")
    print("=" * 70)
    print(f"Model: {args.model_size}")
    print(f"Parameters: {num_params:,}")
    print(f"Max steps: {args.max_steps:,}")
    print(f"Batch size: {args.batch_size} (per device)")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Mixed precision: {args.mixed_precision}")
    print("=" * 70 + "\n")

    # Train
    trainer.train()

    print("\n" + "=" * 70)
    print("BASE PRE-TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print("\nNext steps:")
    print("  1. Evaluate on downstream tasks")
    print("  2. Proceed to Phase 2.2: Mathematical Fine-tuning")
    print("  3. Scale up training with larger datasets")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()