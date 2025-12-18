"""
Phase 2.2: Supervised Fine-Tuning Script

Fine-tune a pre-trained model on AIMO competition problems for
mathematical reasoning.

Usage:
    # Fine-tune from pre-trained checkpoint
    python finetune.py --pretrained checkpoints/pretraining/final.pt

    # Fine-tune with custom config
    python finetune.py --pretrained checkpoints/pretraining/final.pt \
        --learning-rate 2e-5 --epochs 3 --batch-size 8

    # Multi-GPU fine-tuning
    torchrun --nproc_per_node=4 finetune.py --pretrained checkpoints/pretraining/final.pt
"""

import argparse
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Please install torch to run fine-tuning.")
    exit(1)

from src import (
    get_small_config,
    get_base_config,
    get_large_config,
    MathTransformerDecoder,
    MathTokenizer,
)
from src.training import FineTuner, FineTuningConfig, RobustDataCollator
from src.data import (
    AIMODatasetLoader,
    AIMOFormatter,
    AIMOFineTuningDataset,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Phase 2.2: Supervised Fine-Tuning")

    # Model
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["small", "base", "large"],
        help="Model size preset"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained checkpoint"
    )

    # Data
    parser.add_argument(
        "--train-data",
        type=str,
        default="./data/reference.csv",
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio"
    )

    # Training
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Micro batch size per device"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup steps"
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Mixed precision training"
    )

    # Freezing
    parser.add_argument(
        "--freeze-embeddings",
        action="store_true",
        help="Freeze embedding layer"
    )
    parser.add_argument(
        "--freeze-layers",
        type=int,
        default=0,
        help="Number of initial layers to freeze"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/finetuning",
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
        default=500,
        help="Save checkpoint every N steps"
    )

    # Evaluation
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Evaluate every N steps"
    )
    parser.add_argument(
        "--no-generation",
        action="store_true",
        help="Disable solution generation during eval"
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
        default="math-reasoning-finetuning",
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
    """Main fine-tuning function."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    print("\n" + "=" * 70)
    print("PHASE 2.2: SUPERVISED FINE-TUNING")
    print("=" * 70)

    # Check for pretrained checkpoint
    if args.pretrained and not Path(args.pretrained).exists():
        print(f"\n⚠️  Warning: Pretrained checkpoint not found: {args.pretrained}")
        print("Training from scratch (not recommended for fine-tuning)")
        args.pretrained = None

    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = MathTokenizer()
    print(f"✓ Tokenizer initialized (vocab size: {len(tokenizer):,})")

    # Load AIMO dataset
    print(f"\nLoading AIMO dataset from {args.train_data}...")
    loader = AIMODatasetLoader(args.train_data)

    # Split data
    train_problems, val_problems, test_problems = loader.load_split(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
        seed=args.seed,
    )

    print("✓ Dataset loaded:")
    print(f"  Train: {len(train_problems)} problems")
    print(f"  Val:   {len(val_problems)} problems")
    print(f"  Test:  {len(test_problems)} problems")

    # Create formatter
    formatter = AIMOFormatter(use_special_tokens=True, include_reasoning=True)

    # Create datasets
    print("\nCreating PyTorch datasets...")
    train_dataset = AIMOFineTuningDataset(
        problems=train_problems,
        tokenizer=tokenizer,
        formatter=formatter,
        max_length=2048,
    )

    val_dataset = AIMOFineTuningDataset(
        problems=val_problems,
        tokenizer=tokenizer,
        formatter=formatter,
        max_length=2048,
    )

    # Create data loaders with robust collator
    collator = RobustDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_length=2048,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    print("✓ Data loaders created")

    # Initialize model
    print(f"\nInitializing {args.model_size} model...")
    model_config = get_model_config(args.model_size)
    model_config.vocab_size = len(tokenizer)

    model = MathTransformerDecoder(model_config)

    num_params = sum(p.numel() for p in model.parameters())
    print("✓ Model initialized")
    print(f"  Parameters: {num_params:,}")
    print(f"  Size: {num_params * 4 / (1024**2):.2f} MB")

    # Create fine-tuning configuration
    training_config = FineTuningConfig(
        model_config_name=args.model_size,
        pretrained_checkpoint=args.pretrained,
        vocab_size=model_config.vocab_size,
        max_seq_length=2048,
        train_data_path=args.train_data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        micro_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        mixed_precision=args.mixed_precision,
        freeze_embeddings=args.freeze_embeddings,
        freeze_layers=args.freeze_layers,
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.save_interval,
        resume_from_checkpoint=args.resume,
        eval_interval=args.eval_interval,
        generate_samples=not args.no_generation,
        log_interval=args.log_interval,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    print("\n✓ Training configuration created")
    print(f"  Effective batch size: {training_config.effective_batch_size}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Epochs: {args.epochs}")

    # Initialize fine-tuner
    print("\nInitializing Fine-Tuner...")
    finetuner = FineTuner(
        model=model,
        config=training_config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        tokenizer=tokenizer,
        formatter=formatter,
    )

    print("✓ Fine-Tuner ready!")

    # Start fine-tuning
    finetuner.train()

    print("\n" + "=" * 70)
    print("FINE-TUNING COMPLETE!")
    print("=" * 70)
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"Best accuracy: {finetuner.best_accuracy:.4f}")
    print("\nNext steps:")
    print("  1. Evaluate on test set")
    print("  2. Generate solutions for new problems")
    print("  3. Proceed to Phase 2.3: Reinforcement Learning (optional)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()