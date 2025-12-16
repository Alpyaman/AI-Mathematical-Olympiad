# Baseline Training Guide

## Overview

This guide explains how to run baseline training to verify the entire pipeline works end-to-end.

## Quick Start

### 1. Install Training Dependencies

```bash
pip install -r requirements-training.txt
```

Or install PyTorch only:
```bash
pip install torch
```

### 2. Run Baseline Training

```bash
python baseline_training.py
```

This will:
- Create 3 sample problems
- Split into train/val/test
- Train a small model (~22M parameters) for 10 epochs
- Save checkpoints to `./checkpoints/baseline/`
- Show training progress and metrics

### Expected Output

```
======================================================================
BASELINE TRAINING - End-to-End Pipeline Verification
======================================================================

1. Creating sample dataset...
   Created 3 sample problems
   Preprocessed: 3 problems ready
   Split: 2 train, 1 val, 0 test

2. Creating PyTorch datasets...
   Train batches: 2
   Val batches: 1

3. Initializing model...
   Model: 22,609,920 parameters (~86.3MB)
   Config: 4 layers, 256 hidden size

4. Setting up training...
   Optimizer: AdamW (lr=0.001)
   Scheduler: CosineAnnealingLR
   Epochs: 10
   Total steps: 20

5. Training model...

Starting training for 10 epochs...
Device: cuda / cpu
Model parameters: 22,609,920

Epoch 1/10
------------------------------------------------------------
  Step    5 | Loss: 10.8234 | Avg Loss: 10.9156 | LR: 9.88e-04
  Step   10 | Loss: 10.7123 | Avg Loss: 10.8234 | LR: 9.51e-04

  Train Loss: 10.7234
  Val Loss:   10.6543
  Time:       5.2s
  Saved checkpoint to ./checkpoints/baseline/best_model.pt
  New best model! (Val Loss: 10.6543)

...

Training complete! Total time: 0.9 minutes
Best validation loss: 9.2341

======================================================================
BASELINE TRAINING COMPLETE ✓
======================================================================

Training Summary:
  Total epochs: 10
  Total steps: 20
  Best val loss: 9.2341
  Final train loss: 9.3124
  Final val loss: 9.2341

Checkpoints saved to: checkpoints/baseline

Next steps:
  1. Train on larger dataset (MATH, AoPS)
  2. Implement evaluation metrics
  3. Add distributed training support
  4. Proceed to Phase 2: Full Training Infrastructure
```

## What the Script Does

### 1. Data Preparation
- Creates 3 sample mathematical problems
- Applies preprocessing and quality filtering
- Splits into train/validation/test sets
- Creates PyTorch datasets and dataloaders

### 2. Model Initialization
- Uses small configuration (~22M parameters)
- Optimized for quick training on sample data
- Configuration:
  - 4 layers
  - 256 hidden size
  - 4 attention heads
  - 512 max sequence length

### 3. Training Setup
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine annealing learning rate
- **Loss**: Cross-entropy (next-token prediction)
- **Gradient clipping**: Max norm 1.0

### 4. Training Loop
- Trains for 10 epochs
- Logs every 5 steps
- Saves checkpoints every 5 epochs
- Saves best model based on validation loss

### 5. Checkpointing
Checkpoints include:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Training statistics (losses, epoch, step)

## Customization

### Use Larger Model

Edit `baseline_training.py` to use base or large config:

```python
# Instead of get_small_config()
config = get_base_config()  # ~1B parameters
# or
config = get_large_config()  # ~7B parameters
```

### Train on Real Data

Replace sample data with MATH dataset:

```python
# Instead of create_sample_problems()
loader = MathDatasetLoader()
problems = loader.load_math_dataset(split="train")
```

### Adjust Hyperparameters

```python
num_epochs = 50  # More epochs
learning_rate = 5e-4  # Lower learning rate
batch_size = 8  # Larger batches
```

### Enable GPU Training

The script automatically uses GPU if available:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

## Checkpoints

### Loading a Checkpoint

```python
from baseline_training import Trainer

# Load checkpoint
trainer.load_checkpoint("best_model.pt")

# Continue training
trainer.train(num_epochs=10)
```

### Inference with Trained Model

```python
import torch
from src import MathTransformerDecoder, get_small_config

# Load model
config = get_small_config()
model = MathTransformerDecoder(config)

checkpoint = torch.load("checkpoints/baseline/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Generate
input_ids = torch.tensor([[1, 2, 3, 4]])  # Your input
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=100)
```

## Monitoring Training

### Track Loss

The trainer automatically tracks:
- Training loss per epoch
- Validation loss per epoch
- Loss per step (logged)

Access via:
```python
print(trainer.train_losses)
print(trainer.val_losses)
```

### Add TensorBoard (Optional)

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/baseline')

# In training loop
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
```

View with:
```bash
tensorboard --logdir=runs/baseline
```

### Add Weights & Biases (Optional)

```python
import wandb

wandb.init(project="math-reasoning", name="baseline")

# In training loop
wandb.log({"train_loss": train_loss, "val_loss": val_loss})
```

## Troubleshooting

### Out of Memory

Reduce batch size or model size:
```python
batch_size = 1
config.hidden_size = 128
config.num_hidden_layers = 2
```

### Training Too Slow

- Use GPU if available
- Increase batch size
- Reduce max_sequence_length
- Use fewer training steps

### Loss Not Decreasing

- Check learning rate (try 1e-4 to 1e-3)
- Verify data is formatted correctly
- Check for NaN/Inf values
- Try gradient clipping

### CUDA Out of Memory

```python
# Enable gradient checkpointing
config.gradient_checkpointing = True

# Or use smaller model
config = get_small_config()
config.hidden_size = 128
```

## Performance Tips

### Speed Up Training

1. **Use GPU**: 10-100x faster than CPU
2. **Larger batch size**: Better GPU utilization
3. **Mixed precision**: FP16 training
   ```python
   from torch.cuda.amp import autocast, GradScaler
   ```
4. **Gradient accumulation**: Simulate larger batches
   ```python
   accumulation_steps = 4
   ```

### Reduce Memory

1. **Gradient checkpointing**: Trade compute for memory
2. **Smaller batch size**: 1-2 for large models
3. **Smaller max_length**: 256-512 instead of 2048
4. **Model parallelism**: Split model across GPUs

## Next Steps

After baseline training works:

1. **Download MATH Dataset**
   ```bash
   git clone https://github.com/hendrycks/math
   ```

2. **Train on Real Data**
   - Load MATH dataset
   - Increase epochs (50-100)
   - Use base config (~1B parameters)

3. **Implement Evaluation**
   - Answer accuracy
   - Step-by-step correctness
   - Reasoning quality

4. **Scale Up**
   - Multi-GPU training
   - Larger models
   - More data

5. **Proceed to Phase 2**
   - Full training infrastructure
   - Distributed training
   - Advanced evaluation metrics

## Architecture Overview

```
Input (Problem)
    ↓
Tokenizer (200+ math symbols)
    ↓
Embeddings
    ↓
Decoder Layers (4-32)
  - Self-Attention + RoPE
  - Feed-Forward (SwiGLU)
  - RMSNorm
    ↓
Output Projection
    ↓
Loss (Cross-Entropy)
    ↓
Optimizer Step (AdamW)
```

## Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model Size | 22M | Parameters (small config) |
| Layers | 4 | Transformer layers |
| Hidden Size | 256 | Model dimension |
| Attention Heads | 4 | Multi-head attention |
| Batch Size | 1 | Samples per step |
| Learning Rate | 1e-3 | Initial LR |
| Epochs | 10 | Training iterations |
| Max Length | 512 | Tokens per sequence |

## File Structure

```
AI-Mathematical-Olympiad/
├── baseline_training.py          # Main training script
├── checkpoints/
│   └── baseline/
│       ├── best_model.pt         # Best model checkpoint
│       └── checkpoint_epoch_*.pt # Periodic checkpoints
├── src/
│   ├── model/                    # Model architecture
│   ├── tokenizer/                # Mathematical tokenizer
│   └── data/                     # Data pipeline
└── requirements-training.txt     # Training dependencies
```

---

**Status**: ✅ Baseline training script ready
**Next**: Run `python baseline_training.py` on your machine
**Goal**: Verify end-to-end pipeline works correctly