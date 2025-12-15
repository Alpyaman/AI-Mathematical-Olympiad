# 🎓 Complete SLM Training Guide - Summary

## 📁 Files Created

### Core Pipeline Files:
1. **`step1_data_preparation.py`** - Prepare and augment training data
2. **`step2_train_slm.py`** - Train the Small Language Model
3. **`step3_inference_slm.py`** - Generate predictions & submissions
4. **`run_slm_pipeline.py`** - Automated complete pipeline

### Guides:
- **`SLM_COMPLETE_GUIDE.md`** - Detailed documentation
- **`SLM_TRAINING_GUIDE.md`** - Step-by-step instructions

### Legacy/Alternative Solvers:
- `math_solver_hybrid.py` - Symbolic solver (no ML)
- `solver_with_llm.py` - LLM API wrapper (requires API keys)
- `train_math_solver.py` - Traditional ML approach

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data
python step1_data_preparation.py

# 3. Train model (requires GPU or patience for CPU)
python step2_train_slm.py

# 4. Generate submission
python step3_inference_slm.py
```

### Or run everything automatically:
```bash
python run_slm_pipeline.py
```

---

## 🎯 What Each Step Does

### Step 1: Data Preparation
**Input:** `data/reference.csv` (10 olympiad problems)
**Process:**
- Generates 200 simple training problems
- Augments to 400+ examples
- Creates train/val splits
**Output:** `data/train.jsonl`, `data/val.jsonl`

### Step 2: Training
**Input:** Training data + Base model (Phi-2)
**Process:**
- Applies LoRA for efficient fine-tuning
- Trains for 3 epochs
- Saves checkpoints
**Output:** `models/math_slm/` (trained model)
**Time:** 2-6 hours on GPU, 12-24 hours on CPU

### Step 3: Inference
**Input:** Trained model + `data/test.csv`
**Process:**
- Generates solutions for each problem
- Extracts numerical answers
**Output:** `submission_slm.csv`

---

## 💻 Hardware Requirements

### Minimum (CPU Only):
- 16GB RAM
- 20GB disk space
- Time: 12-24 hours training

### Recommended (GPU):
- NVIDIA GPU with 8-16GB VRAM
- 16GB RAM
- 20GB disk space
- Time: 2-6 hours training

### Cloud Options (Free):
- ✅ **Google Colab** (Free T4 GPU)
- ✅ **Kaggle Notebooks** (Free P100 GPU)
- ✅ **Paperspace Gradient** (Free tier)

---

## 📊 Expected Performance

### Simple Problems (test.csv):
```
Problem: What is $1-1$?
Expected: 0
Symbolic solver: ✅ 100% accurate
Trained SLM: ✅ 95-100% accurate
```

### Complex Olympiad Problems (reference.csv):
```
Problem: Complex geometry with circles and triangles...
Expected: 336
Symbolic solver: ❌ 0% (too complex)
Trained SLM (minimal data): ⚠️ 5-15% accurate
Trained SLM (with GPT-4 data): ✅ 30-50% accurate
```

---

## 🎨 Model Options

### Recommended: microsoft/phi-2 (2.7B)
- ✅ Best balance of size/performance
- ✅ Runs in 8GB VRAM
- ✅ Good at reasoning
- Training time: 3-4 hours (GPU)

### Alternative: TinyLlama (1.1B)
- ✅ Smaller, faster
- ✅ Runs in 6GB VRAM  
- ⚠️ Less capable
- Training time: 2-3 hours (GPU)

### Smallest: pythia-410m (410M)
- ✅ Very fast
- ✅ Runs in 4GB VRAM
- ⚠️ Limited reasoning
- Training time: 1-2 hours (GPU)

---

## 🔧 Customization

### Change Model:
```python
# In step2_train_slm.py, line ~xxx
trainer = SLMTrainer(
    model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',  # Change here
    output_dir='models/math_slm',
    use_lora=True
)
```

### More Training Data:
```python
# In step1_data_preparation.py
preparator.create_simple_problems(num_examples=1000)  # More examples
preparator.augment_data(augmentation_factor=5)  # More augmentation
```

### Longer Training:
```python
# In step2_train_slm.py
trainer.train(
    num_epochs=10,  # More epochs
    learning_rate=1e-5  # Lower LR for stability
)
```

---

## 🐛 Troubleshooting

### "Out of memory" during training:
```python
# Solution 1: Reduce batch size
batch_size=1

# Solution 2: Use smaller model
model_name='EleutherAI/pythia-410m'

# Solution 3: Enable gradient checkpointing
model.gradient_checkpointing_enable()
```

### "CUDA not available":
```python
# Train on CPU (slower)
device='cpu'

# Or use free GPU:
# - Google Colab: https://colab.research.google.com
# - Kaggle: https://kaggle.com/notebooks
```

### "Model not converging":
```python
# Reduce learning rate
learning_rate=1e-5

# More training data
create_simple_problems(num_examples=500)

# Train longer
num_epochs=10
```

---

## 📦 Deployment to Kaggle

### Option 1: Upload Model as Dataset
```bash
# 1. Create Kaggle dataset
kaggle datasets create -p models/math_slm

# 2. In notebook, load model
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('/kaggle/input/your-model')
```

### Option 2: Include Code in Notebook
```python
# Copy step3_inference_slm.py to notebook
# Add model files to notebook
# Run inference directly
```

---

## 🎯 Competition Strategy

### Phase 1: Baseline (Current)
- ✅ Symbolic solver for simple problems
- ⚠️ Random/0 for complex problems
- Expected: ~50% score

### Phase 2: Trained SLM
- ✅ Symbolic solver for simple
- ✅ Trained SLM for moderate
- ⚠️ Fallback for complex
- Expected: ~60-70% score

### Phase 3: Enhanced SLM
- ✅ More training data (1000+ examples)
- ✅ GPT-4 generated solutions
- ✅ Larger model (7B)
- Expected: ~75-85% score

### Phase 4: Ensemble
- ✅ Multiple trained models
- ✅ Voting/averaging
- ✅ LLM fallback for hardest
- Expected: ~85-90% score

---

## ✅ Checklist

- [ ] Data preparation complete
- [ ] Model training started
- [ ] Training completed successfully
- [ ] Model evaluated on validation set
- [ ] Submission generated
- [ ] Submission uploaded to Kaggle
- [ ] Considered improvements:
  - [ ] More training data
  - [ ] Better base model
  - [ ] Ensemble methods
  - [ ] LLM fallback for hard problems

---

## 📚 Additional Resources

- **Hugging Face Transformers:** https://huggingface.co/docs/transformers
- **LoRA Paper:** https://arxiv.org/abs/2106.09685
- **Math Word Problems:** https://paperswithcode.com/task/math-word-problem-solving
- **Phi-2 Model:** https://huggingface.co/microsoft/phi-2

---

## 🏆 Good Luck!

You now have a complete, self-contained SLM training pipeline that:
- ✅ Requires no external APIs
- ✅ Works offline
- ✅ Can be deployed to Kaggle
- ✅ Is customizable and extensible

**Next step:** Run `python run_slm_pipeline.py` and start training! 🚀
