# 🧮 AI Mathematical Olympiad Project

**A custom transformer model trained to solve International Mathematical Olympiad (IMO) problems**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.9](https://img.shields.io/badge/PyTorch-2.9-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📖 Overview

This project implements a specialized transformer model for mathematical reasoning at the Olympiad level. The model uses:
- **Custom decoder-only architecture** based on Llama/GPT design
- **Mathematical tokenizer** supporting 100+ mathematical symbols (∀, ∃, ∑, ∫, etc.)
- **RoPE positional encoding** for length generalization
- **Modern optimizations**: RMSNorm, SwiGLU activation, Grouped-Query Attention

### 🎯 Project Goals

1. Solve International Mathematical Olympiad (IMO) level problems
2. Generate step-by-step mathematical proofs and solutions
3. Handle complex multi-step reasoning with mathematical rigor
4. Compete in mathematical AI competitions (e.g., Kaggle AIMO)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- 16GB+ RAM (for inference)
- GPU with 8GB+ VRAM (for training) - Use Google Colab if not available
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd AI-Mathematical-Olympiad

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Quick Test

```bash
# Test the architecture
python phase1_1_demo.py

# Run baseline training (CPU, small model, 3 samples)
python baseline_training.py
```

---

## 📁 Project Structure

```
AI-Mathematical-Olympiad/
├── src/                          # Core implementation
│   ├── model/                    # Transformer architecture
│   │   ├── decoder.py           # Main model implementation
│   │   └── rope.py              # Rotary positional embeddings
│   ├── tokenizer/               # Mathematical tokenizer
│   │   └── math_tokenizer.py   # Custom tokenizer with math symbols
│   ├── data/                    # Data loading and preprocessing
│   ├── training/                # Training utilities
│   ├── evaluation/              # Evaluation metrics
│   └── config/                  # Model configurations
│
├── data/                        # Training and test data
│   ├── reference.csv           # Reference problems
│   ├── test.csv                # Test problems
│   └── cache/                  # Preprocessed data cache
│
├── checkpoints/                 # Saved model checkpoints
│   ├── baseline/               # Baseline training results
│   └── pretraining_notebook/   # Colab training results
│
├── notebooks/                   # Jupyter notebooks (implied by .ipynb files)
│   ├── colab_train_full_dataset.ipynb
│   └── phase2_1_pretraining.ipynb
│
├── Training Scripts
│   ├── baseline_training.py    # Quick pipeline verification
│   ├── train_math_real.py      # Main training script
│   ├── pretrain.py             # Pre-training script
│   └── finetune.py             # Fine-tuning script
│
└── Documentation
    ├── PHASE1_1_README.md      # Architecture details
    ├── PHASE1_2_README.md      # Data preparation
    ├── PHASE2_1_README.md      # Training details
    └── BASELINE_TRAINING.md    # Baseline training guide
```

---

## 🎓 Training Pipeline

### Phase 1: Foundation ✅

**Status**: Complete

- ✅ Custom transformer architecture implemented
- ✅ Mathematical tokenizer with 100+ symbols
- ✅ Data preprocessing pipeline
- ✅ Multiple model sizes (125M, 1B, 7B parameters)

### Phase 2: Training ⚠️

**Status**: In Progress

#### Option 1: Local Training (Baseline)
```bash
# Quick verification (CPU, 3 samples, 10 epochs)
python baseline_training.py
```

#### Option 2: Google Colab Training (Recommended)
```bash
# 1. Open colab_train_full_dataset.ipynb in Google Colab
# 2. Runtime → Change runtime type → GPU (T4)
# 3. Run all cells
# 4. Training time: ~2-4 hours on T4 GPU
```

#### Option 3: Full Training (Requires GPU)
```bash
# Train on MATH dataset (~7,500 problems)
python train_math_real.py

# Configuration
# - Model: Small config (125M params) or Base (1B params)
# - Dataset: MATH dataset via Hugging Face
# - Training: Mixed precision, gradient accumulation
# - Checkpoints: Saved every 5 epochs
```

### Phase 3: Evaluation ⚠️

**Status**: Partially Complete

```bash
# Evaluate trained model
python test_inference.py

# Debug inference
python debug_inference.py
```

---

## 📊 Model Configurations

### Small (125M parameters) - **Recommended for Testing**
- Hidden size: 512
- Layers: 8
- Attention heads: 8
- Context length: 2048
- Training time: ~2 hours on T4 GPU
- Memory: ~2GB GPU VRAM

### Base (1B parameters) - **Recommended for Competition**
- Hidden size: 2048
- Layers: 24
- Attention heads: 16
- Context length: 8192
- Training time: ~12 hours on T4 GPU
- Memory: ~16GB GPU VRAM

### Large (7B parameters) - **Research Grade**
- Hidden size: 4096
- Layers: 32
- Attention heads: 32 (8 KV heads with GQA)
- Context length: 16384
- Training time: ~48+ hours on A100 GPU
- Memory: ~112GB GPU VRAM (requires multi-GPU)

---

## 🧪 Testing & Inference

### Test the Tokenizer
```bash
python test_enhanced_tokenizer.py
```

### Test Model Generation
```bash
python test_inference.py
```

### Check Token Counts
```bash
python check_tokens.py
```

---

## 📈 Current Progress

- [x] Phase 1.1: Architecture Design
- [x] Phase 1.2: Data Acquisition
- [x] Phase 2.3: Base Pre-training (Partially - checkpoints exist)
- [ ] Phase 2.4: Supervised Fine-tuning (In Progress)
- [x] Phase 2.5: Training Stability Optimizations
- [ ] Phase 3.6: RLHF/DPO (Future)
- [ ] Phase 3.7: Robust Evaluation (Partially Complete)

---

## 🎯 Next Steps

### Immediate (This Week)
1. ✅ Complete baseline training verification
2. ⚠️ Train on full MATH dataset via Colab
3. ⚠️ Implement comprehensive evaluation metrics
4. ⚠️ Test on Kaggle competition problems

### Short-term (Next 2-4 Weeks)
1. Fine-tune on Olympiad-specific problems
2. Implement answer extraction and verification
3. Optimize for competition submission format
4. Add ensemble methods if beneficial

### Long-term (Future)
1. Implement RLHF/DPO for better alignment
2. Scale to larger model sizes
3. Add external tool integration (symbolic solver)
4. Evaluate on contamination-free benchmarks

---

## 📚 Resources & References

### Datasets
- [MATH Dataset](https://github.com/hendrycks/math) - 12,500 competition problems
- [AoPS Dataset](https://artofproblemsolving.com/) - Art of Problem Solving community
- Kaggle AIMO Competition Data

### Papers & Architecture
- [Llama 2](https://arxiv.org/abs/2307.09288) - Base architecture inspiration
- [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864) - Rotary positional embeddings
- [GLU Variants](https://arxiv.org/abs/2002.05202) - SwiGLU activation
- [RMSNorm](https://arxiv.org/abs/1910.07467) - Efficient normalization

---

## 🤝 Contributing

This is a learning/competition project. Feel free to:
- Report issues
- Suggest improvements
- Share training results
- Contribute evaluation metrics

---

## 📄 License

[Specify your license here - MIT recommended for open source]

---

## 🙏 Acknowledgments

- Hugging Face for Transformers library
- Google Colab for free GPU access
- MATH dataset creators
- Mathematical reasoning research community

---

## 📞 Contact

[Your contact information or links]

---

**Last Updated**: December 2024
**Status**: Active Development - Phase 2 (Training)
