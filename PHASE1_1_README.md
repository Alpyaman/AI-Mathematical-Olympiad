# Phase 1.1: Initial Architecture Design

## Overview

Phase 1.1 establishes the foundation for the AI Mathematical Olympiad project by implementing a customized decoder-only transformer architecture optimized for mathematical reasoning.

## Architecture Highlights

### 🏗️ Decoder-Only Transformer
- Based on proven Llama/GPT architecture
- **Multi-Head Attention** with Grouped-Query Attention (GQA) for efficiency
- **SwiGLU** activation in feed-forward networks
- **RMSNorm** for faster and more stable training
- Pre-normalization architecture

### 📏 Positional Encoding
- **Rotary Position Embeddings (RoPE)** for better position-aware attention
- **Dynamic RoPE Scaling** for length generalization
- Supports extrapolation to 2-4x the training sequence length
- Essential for handling variable-length mathematical proofs

### 📖 Extended Context Length
- **Default**: 8192 tokens
- **Small config**: 2048 tokens
- **Large config**: 16384 tokens
- Sufficient for multi-step mathematical proofs and complex reasoning chains

### 🔤 Mathematical Tokenizer
- Handles **100+ mathematical symbols**: ∀, ∃, ∑, ∫, ∈, ⊂, ≤, ≥, etc.
- **LaTeX notation support**: `\forall`, `\sum`, `\int`, etc.
- **Special tokens** for mathematical structures:
  - `<math>`, `</math>` for mathematical expressions
  - `<proof>`, `</proof>` for proof sections
  - `<step>` for reasoning steps
- Character-level fallback for unknown symbols

## Directory Structure

```
src/
├── __init__.py                 # Main package exports
├── config/
│   ├── __init__.py
│   └── model_config.py         # Model configurations
├── model/
│   ├── __init__.py
│   ├── decoder.py              # Main transformer decoder
│   └── rope.py                 # RoPE implementations
└── tokenizer/
    ├── __init__.py
    └── math_tokenizer.py       # Mathematical tokenizer
```

## Model Configurations

### Small Config (~125M parameters)
- Hidden size: 512
- Layers: 8
- Attention heads: 8
- Context length: 2048
- **Use case**: Quick experimentation, testing

### Base Config (~1B parameters)
- Hidden size: 2048
- Layers: 24
- Attention heads: 16
- Context length: 8192
- **Use case**: Main training, balanced performance

### Large Config (~7B parameters)
- Hidden size: 4096
- Layers: 32
- Attention heads: 32 (8 KV heads with GQA)
- Context length: 16384
- **Use case**: Maximum performance, complex problems

## Quick Start

### Installation

```bash
# Install dependencies
pip install torch numpy

# The architecture is self-contained in the src/ directory
```

### Basic Usage

```python
from src import (
    MathTransformerConfig,
    MathTransformerDecoder,
    MathTokenizer,
    get_base_config,
)

# Initialize tokenizer
tokenizer = MathTokenizer()

# Create model with base configuration
config = get_base_config()
model = MathTransformerDecoder(config)

# Tokenize mathematical text
text = "Prove that ∀n ∈ ℕ, n² + n is even."
encoded = tokenizer.encode(text)

# Forward pass
import torch
input_ids = torch.tensor([encoded["input_ids"]])
outputs = model(input_ids=input_ids)
logits = outputs["logits"]
```

### Run Demo

```bash
python phase1_1_demo.py
```

The demo showcases:
1. Mathematical tokenizer capabilities
2. Model architecture and parameter count
3. Forward pass through the model
4. Text generation (with untrained model)
5. RoPE length generalization

## Key Features

### 1. Length Generalization

The dynamic RoPE implementation allows the model to handle sequences longer than those seen during training:

```python
from src.model.rope import DynamicRoPE

rope = DynamicRoPE(
    dim=64,
    max_position_embeddings=2048,
    max_scaling_factor=4.0,  # Can handle up to 8192 tokens
)
```

### 2. Efficient Inference

KV caching reduces computational cost for autoregressive generation:

```python
# Generate with KV caching
generated = model.generate(
    input_ids=input_ids,
    max_new_tokens=512,
    use_cache=True,
)
```

### 3. Grouped-Query Attention

Reduces memory usage while maintaining performance:

```python
config = MathTransformerConfig(
    num_attention_heads=32,
    num_key_value_heads=8,  # 4x fewer KV heads
)
```

## Mathematical Notation Support

The tokenizer handles various mathematical notations:

| Symbol | LaTeX | Description |
|--------|-------|-------------|
| ∀ | `\forall` | For all (universal quantifier) |
| ∃ | `\exists` | There exists (existential quantifier) |
| ∈ | `\in` | Element of |
| ⊂ | `\subset` | Subset |
| ∑ | `\sum` | Summation |
| ∫ | `\int` | Integral |
| ≤ | `\leq` | Less than or equal |
| ≥ | `\geq` | Greater than or equal |
| ∞ | `\infty` | Infinity |
| π | `\pi` | Pi |

And many more! See `src/tokenizer/math_tokenizer.py` for the complete list.

## Design Decisions

### Why Decoder-Only?

1. **Proven Performance**: GPT and Llama architectures have shown excellent results
2. **Autoregressive Generation**: Natural fit for step-by-step mathematical reasoning
3. **Simplicity**: Easier to train and optimize than encoder-decoder models
4. **Scalability**: Well-studied scaling properties

### Why RoPE?

1. **Length Generalization**: Better extrapolation to longer sequences
2. **Relative Positions**: Captures relative positional relationships
3. **No Absolute Position Limits**: Can theoretically handle infinite sequences
4. **Proven Effectiveness**: Used successfully in Llama, PaLM, and other models

### Why RMSNorm?

1. **Faster**: ~10-15% faster than LayerNorm
2. **Simpler**: No mean centering, only scaling
3. **Effective**: Works as well as LayerNorm in practice
4. **Lower Memory**: Reduced memory footprint

## Implementation Details

### Attention Mechanism

```python
# Multi-head attention with RoPE
query, key = rope(query, key, position_ids)
scores = query @ key.T / sqrt(head_dim)
scores = scores + causal_mask
attention = softmax(scores) @ value
```

### Feed-Forward Network (SwiGLU)

```python
# SwiGLU activation
gate = silu(gate_proj(x)) * up_proj(x)
output = down_proj(gate)
```

### Training Optimizations

- **Gradient Checkpointing**: Reduce memory usage for large models
- **Mixed Precision**: Support for FP16/BF16 training
- **Flash Attention Ready**: Can integrate Flash Attention 2 for speed

## Next Steps (Phase 1.2)

The architecture is ready for Phase 1.2: Dataset Preparation

1. **Data Collection**
   - Mathematical olympiad problems (IMO, USAMO, etc.)
   - Proof datasets
   - Mathematical reasoning datasets

2. **Data Preprocessing**
   - Convert problems to tokenizer format
   - Create training/validation splits
   - Data augmentation strategies

3. **Training Setup**
   - Training loop implementation
   - Learning rate scheduling
   - Evaluation metrics

4. **Baseline Training**
   - Train on small dataset
   - Establish baseline performance
   - Iterate on architecture if needed

## Performance Considerations

### Memory Requirements (FP32)

| Config | Parameters | Memory (Model) | Memory (Training) |
|--------|-----------|----------------|-------------------|
| Small  | ~125M     | ~500 MB        | ~2 GB            |
| Base   | ~1B       | ~4 GB          | ~16 GB           |
| Large  | ~7B       | ~28 GB         | ~112 GB          |

*Training memory includes gradients, optimizer states, and activations*

### Recommended Hardware

- **Small**: 1x GPU with 8GB+ VRAM (RTX 3070, etc.)
- **Base**: 1x GPU with 24GB+ VRAM (RTX 4090, A5000, etc.)
- **Large**: 4-8x GPUs with 40GB+ VRAM (A100, H100, etc.)

## References

1. **Llama 2**: [Touvron et al., 2023](https://arxiv.org/abs/2307.09288)
2. **RoFormer (RoPE)**: [Su et al., 2021](https://arxiv.org/abs/2104.09864)
3. **GLU Variants**: [Shazeer, 2020](https://arxiv.org/abs/2002.05202)
4. **RMSNorm**: [Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)

## License

This implementation is part of the AI Mathematical Olympiad project.

---

**Phase 1.1 Status**: ✅ Complete

**Ready for**: Phase 1.2 - Dataset Preparation