# Phase 2.1: Base Pre-Training

## Overview

Phase 2.1 implements base pre-training infrastructure for training the mathematical reasoning model on large-scale corpora. This phase focuses on building general language understanding and preliminary mathematical fluency through causal language modeling on a mixture of mathematical documents (ArXiv papers) and general text.

## Architecture

### Key Components

1. **Streaming Data Pipeline**
   - `TextStreamDataset`: Memory-efficient streaming dataset for large corpora
   - `MixedDomainDataset`: Samples from multiple domains with configurable weights
   - Supports JSONL format for easy data processing
   - Dynamic batching with padding

2. **Distributed Training Infrastructure**
   - PyTorch DistributedDataParallel (DDP) support
   - Automatic process group initialization
   - Gradient synchronization across GPUs
   - Efficient checkpoint saving/loading

3. **Training Optimizations**
   - Mixed precision training (fp16/bf16)
   - Gradient accumulation for large effective batch sizes
   - Gradient checkpointing to reduce memory
   - Learning rate scheduling (warmup + cosine decay)
   - Gradient clipping for stability

4. **Monitoring and Logging**
   - TensorBoard integration
   - Weights & Biases (wandb) support
   - Comprehensive metrics tracking
   - Automatic checkpointing with cleanup

## File Structure

```
src/
├── training/
│   ├── __init__.py              # Training module exports
│   ├── config.py                # PreTrainingConfig dataclass
│   ├── distributed.py           # Distributed training utilities
│   └── pretrainer.py            # PreTrainer class
│
├── data/
│   └── pretraining_dataset.py   # Streaming datasets
│
pretrain.py                       # Main pre-training script
phase2_1_demo.py                 # Demo showcasing infrastructure
```

## Usage

### Quick Start (Demo)

Run a quick demo to verify the infrastructure:

```bash
python phase2_1_demo.py
```

This will:
- Create sample data (mathematical + general text)
- Initialize the streaming dataset with mixed sampling
- Set up the pre-trainer with all optimizations
- Run 20 training steps to verify everything works

### Data Preparation

Create your pre-training data in JSONL format:

```bash
data/pretraining/
├── arxiv/
│   ├── papers_001.jsonl
│   ├── papers_002.jsonl
│   └── ...
└── general/
    ├── c4_001.jsonl
    ├── wikipedia_001.jsonl
    └── ...
```

Each JSONL line should contain:
```json
{"text": "Your document text here..."}
```

Or use the helper to create sample data:

```python
from src.data.pretraining_dataset import create_sample_pretraining_data
create_sample_pretraining_data("./data/pretraining")
```

### Single GPU Training

```bash
python pretrain.py \
    --model-size small \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --max-steps 100000 \
    --learning-rate 3e-4 \
    --mixed-precision bf16
```

### Multi-GPU Training (Distributed)

```bash
torchrun --nproc_per_node=4 pretrain.py \
    --model-size base \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --max-steps 500000 \
    --learning-rate 3e-4 \
    --mixed-precision bf16 \
    --use-wandb
```

### Resume from Checkpoint

```bash
python pretrain.py \
    --resume checkpoints/pretraining/step_50000.pt \
    --max-steps 100000
```

## Configuration

### PreTrainingConfig

Key configuration options:

```python
from src.training import PreTrainingConfig

config = PreTrainingConfig(
    # Model
    model_config_name="base",         # small, base, or large
    max_seq_length=2048,

    # Data
    data_dir="./data/pretraining",
    data_sources=["arxiv", "general"],
    data_mix_weights=[0.3, 0.7],      # 30% math, 70% general

    # Training
    micro_batch_size=4,               # Per-device batch size
    gradient_accumulation_steps=8,    # Effective batch = 32
    max_steps=100000,
    warmup_steps=2000,
    learning_rate=3e-4,
    min_learning_rate=3e-5,

    # Optimization
    mixed_precision="bf16",           # fp16, bf16, or fp32
    gradient_checkpointing=True,
    max_grad_norm=1.0,

    # Checkpointing
    checkpoint_dir="./checkpoints/pretraining",
    save_interval=5000,
    keep_last_n_checkpoints=3,

    # Logging
    log_interval=10,
    use_wandb=True,
    use_tensorboard=True,
)
```

## Training Features

### Mixed-Domain Sampling

The pre-training pipeline samples from multiple data sources with configurable weights:

```python
data_sources = ["arxiv", "general"]
data_mix_weights = [0.3, 0.7]  # 30% mathematical, 70% general
```

This ensures the model learns:
- **Mathematical fluency** from ArXiv papers (proofs, theorems, equations)
- **General language** from C4, Wikipedia, books

### Streaming Data

For datasets too large to fit in memory:

```python
from src.data.pretraining_dataset import TextStreamDataset

dataset = TextStreamDataset(
    data_paths=["data/large_corpus.jsonl"],
    tokenizer=tokenizer,
    max_seq_length=2048,
    buffer_size=10000,     # Shuffle buffer
    shuffle_buffer=True,
)
```

### Gradient Accumulation

Achieve large effective batch sizes on limited hardware:

```python
# Effective batch size = micro_batch_size * gradient_accumulation_steps * num_gpus
# Example: 4 * 8 * 4 = 128
config = PreTrainingConfig(
    micro_batch_size=4,
    gradient_accumulation_steps=8,
)
```

### Mixed Precision

Faster training with less memory:

```python
config = PreTrainingConfig(
    mixed_precision="bf16",  # Recommended for modern GPUs
    # or "fp16" for older GPUs
)
```

### Learning Rate Schedule

Warmup + Cosine decay:

```
LR
│     ╱────╲
│    ╱      ╲___
│   ╱           ╲____
│  ╱                  ╲____
│ ╱                        ╲
└─────────────────────────────> Steps
  warmup     cosine decay
```

## Monitoring

### TensorBoard

```bash
tensorboard --logdir runs/pretraining
```

Tracks:
- Training loss
- Learning rate
- Tokens per second
- Gradient norms

### Weights & Biases

```bash
python pretrain.py --use-wandb --wandb-project my-project
```

Provides:
- Real-time metrics
- System monitoring
- Model checkpoints
- Experiment comparison

## Checkpointing

Checkpoints are automatically saved every `save_interval` steps:

```
checkpoints/pretraining/
├── step_5000.pt
├── step_10000.pt
├── step_15000.pt
├── best.pt              # Best validation loss
└── final.pt             # Final checkpoint
```

Each checkpoint contains:
- Model state dict
- Optimizer state
- Scheduler state
- Training step
- Tokens seen
- Configuration

## Performance Tips

### Memory Optimization

1. **Gradient Checkpointing**: Trades compute for memory
   ```python
   config.gradient_checkpointing = True
   ```

2. **Mixed Precision**: Reduces memory by 2x
   ```python
   config.mixed_precision = "bf16"
   ```

3. **Smaller Batch Size**: Reduce micro_batch_size, increase gradient accumulation
   ```python
   config.micro_batch_size = 2
   config.gradient_accumulation_steps = 16
   ```

### Speed Optimization

1. **Larger Batch Size**: If memory allows
   ```python
   config.micro_batch_size = 8
   ```

2. **Multiple GPUs**: Use distributed training
   ```bash
   torchrun --nproc_per_node=8 pretrain.py
   ```

3. **Efficient Data Loading**:
   ```python
   config.num_workers = 4
   config.pin_memory = True
   ```

## Data Sources

### Recommended Mathematical Corpora

1. **ArXiv Papers**
   - Download: https://arxiv.org/help/bulk_data
   - Extract text from LaTeX/PDF
   - ~2M papers in math/CS

2. **Proof Corpus**
   - Lean mathlib
   - Isabelle/HOL
   - Metamath

3. **Math Textbooks**
   - OpenStax
   - LibreTexts
   - Project Gutenberg

### Recommended General Text

1. **C4 (Colossal Clean Crawled Corpus)**
   - 750GB of cleaned web text
   - Download: https://huggingface.co/datasets/c4

2. **Wikipedia**
   - Download dumps: https://dumps.wikimedia.org/
   - ~6M articles in English

3. **Books**
   - Project Gutenberg
   - BookCorpus

## Example: Data Preprocessing

Convert ArXiv LaTeX to JSONL:

```python
import json
import glob

def process_arxiv_papers(input_dir, output_file):
    """Convert ArXiv LaTeX files to JSONL."""
    with open(output_file, 'w', encoding='utf-8') as out:
        for latex_file in glob.glob(f"{input_dir}/**/*.tex", recursive=True):
            with open(latex_file, 'r', encoding='utf-8') as f:
                text = f.read()

            # Clean LaTeX (remove commands, etc.)
            text = clean_latex(text)

            # Write JSONL
            out.write(json.dumps({"text": text}) + "\n")

def clean_latex(text):
    """Basic LaTeX cleaning."""
    # Remove comments
    text = re.sub(r'%.*$', '', text, flags=re.MULTILINE)

    # Remove common commands
    text = re.sub(r'\\[a-zA-Z]+', '', text)

    # Keep math symbols
    return text
```

## Training Recommendations

### For Small Models (<1B params)

```bash
python pretrain.py \
    --model-size small \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --max-steps 100000 \
    --learning-rate 3e-4 \
    --mixed-precision bf16
```

### For Base Models (1-7B params)

```bash
torchrun --nproc_per_node=4 pretrain.py \
    --model-size base \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --max-steps 500000 \
    --learning-rate 3e-4 \
    --mixed-precision bf16
```

### For Large Models (>7B params)

```bash
torchrun --nproc_per_node=8 pretrain.py \
    --model-size large \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --max-steps 1000000 \
    --learning-rate 2e-4 \
    --mixed-precision bf16 \
    --gradient-checkpointing
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size:
   ```bash
   --batch-size 2
   ```

2. Enable gradient checkpointing:
   ```bash
   python pretrain.py  # Already enabled by default
   ```

3. Use mixed precision:
   ```bash
   --mixed-precision bf16
   ```

### Slow Training

1. Increase batch size if memory allows
2. Use multiple GPUs
3. Increase num_workers for data loading
4. Use SSD for data storage

### NaN Loss

1. Lower learning rate:
   ```bash
   --learning-rate 1e-4
   ```

2. Increase gradient clipping:
   ```python
   config.max_grad_norm = 0.5
   ```

3. Use bf16 instead of fp16 (more stable)

## Next Steps

After completing base pre-training:

1. **Evaluate on downstream tasks**
   - Math problem solving
   - Proof generation
   - Symbolic reasoning

2. **Phase 2.2: Mathematical Fine-tuning**
   - Fine-tune on MATH dataset
   - Fine-tune on formal proofs
   - Reinforce with outcome supervision

3. **Scale up training**
   - Larger model (base → large)
   - More data (100B+ tokens)
   - Longer training (1M+ steps)

4. **Advanced techniques**
   - Curriculum learning
   - Data filtering and deduplication
   - Multi-stage training

## References

- [Chinchilla: Training Compute-Optimal LLMs](https://arxiv.org/abs/2203.15556)
- [LLaMA: Open and Efficient Foundation LLMs](https://arxiv.org/abs/2302.13971)
- [Minerva: Solving Quantitative Reasoning Problems](https://arxiv.org/abs/2206.14858)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

---

**Phase 2.1 Status**: ✅ Complete

Training infrastructure ready for large-scale pre-training on mixed mathematical and general text corpora.