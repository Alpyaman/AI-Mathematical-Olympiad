# Complete SLM Training Pipeline for Mathematical Olympiad

## 📋 Overview

This pipeline trains a custom Small Language Model (SLM) to solve mathematical olympiad problems without external API dependencies, perfect for Kaggle competitions.

## 🎯 Pipeline Steps

### **Step 1: Data Preparation** 
File: `step1_data_preparation.py`

**What it does:**
- Loads reference problems with answers
- Generates 200 simple training problems (arithmetic, algebra, modular math)
- Augments data by 2x with variations
- Creates train/validation splits (85%/15%)
- Formats data in multiple training formats (JSONL, Alpaca, Conversational)

**Run:**
```bash
python step1_data_preparation.py
```

**Output:**
- `data/train.jsonl` - Training data (~400+ examples)
- `data/val.jsonl` - Validation data
- `data/train_alpaca.jsonl` - Alpaca format

---

### **Step 2: Model Training**
File: `step2_train_slm.py`

**What it does:**
- Loads a small pre-trained model (Phi-2, TinyLlama, or Pythia)
- Applies LoRA for parameter-efficient fine-tuning
- Trains on mathematical problems
- Saves the fine-tuned model

**Supported Models:**
- `microsoft/phi-2` (2.7B params) - **Recommended**
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B params)
- `EleutherAI/pythia-410m` (410M params) - Faster, less capable

**Training Configuration:**
- LoRA rank: 16
- Learning rate: 2e-5
- Epochs: 3
- Batch size: 2 (with gradient accumulation)
- Mixed precision: FP16 (if GPU available)

**Run:**
```bash
python step2_train_slm.py
```

**Output:**
- `models/math_slm/` - Trained model checkpoint

**Hardware Requirements:**
- **GPU (Recommended):** NVIDIA with 8-16GB VRAM
  - Colab free tier: ✓ (T4 GPU)
  - Kaggle notebooks: ✓ (P100 GPU)
- **CPU:** Possible but very slow (4-24 hours)

---

### **Step 3: Inference & Submission**
File: `step3_inference_slm.py`

**What it does:**
- Loads trained model
- Generates solutions for test problems
- Extracts numerical answers
- Creates submission CSV

**Run:**
```bash
# Generate submission
python step3_inference_slm.py

# Interactive testing
python step3_inference_slm.py --interactive
```

**Output:**
- `submission_slm.csv` - Kaggle submission file

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
python step1_data_preparation.py
```

### 3. Train Model (requires GPU)
```bash
python step2_train_slm.py
```

### 4. Generate Predictions
```bash
python step3_inference_slm.py
```

---

## 📊 Training on Different Platforms

### Google Colab (Free GPU)
```python
# Upload your code to Colab
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install -q transformers peft accelerate bitsandbytes

# Run training
!python step2_train_slm.py
```

### Kaggle Notebooks (Free GPU)
1. Create new notebook
2. Enable GPU (Settings → Accelerator → GPU)
3. Add dataset with reference.csv
4. Upload code files
5. Run training pipeline

---

## 🎛️ Customization Options

### Change Base Model
Edit `step2_train_slm.py`:
```python
trainer = SLMTrainer(
    model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',  # Change this
    output_dir='models/math_slm',
    use_lora=True
)
```

### Adjust Training
```python
trainer.train(
    num_epochs=5,           # More epochs for better learning
    batch_size=1,           # Reduce if out of memory
    learning_rate=1e-5,     # Lower for more stable training
    save_steps=50           # Save more frequently
)
```

### Data Augmentation
Edit `step1_data_preparation.py`:
```python
preparator.augment_data(augmentation_factor=5)  # More augmentation
preparator.create_simple_problems(num_examples=500)  # More examples
```

---

## 📦 Model Deployment for Kaggle

### Option A: Direct Deployment
1. Train model
2. Upload `models/math_slm/` folder to Kaggle dataset
3. Load in submission notebook:
```python
from step3_inference_slm import MathSLMInference
solver = MathSLMInference('/kaggle/input/your-model/math_slm')
```

### Option B: Quantization (Smaller Size)
```python
# After training, quantize model
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained('models/math_slm')
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
model.save_pretrained('models/math_slm_quantized')
```

### Option C: ONNX Export (Faster Inference)
```python
from optimum.onnxruntime import ORTModelForCausalLM

ort_model = ORTModelForCausalLM.from_pretrained(
    'models/math_slm',
    export=True
)
ort_model.save_pretrained('models/math_slm_onnx')
```

---

## 🔍 Evaluation & Testing

### Test on Reference Problems
```python
from step3_inference_slm import MathSLMInference
import pandas as pd

solver = MathSLMInference('models/math_slm')
reference = pd.read_csv('data/reference.csv')

correct = 0
for _, row in reference.iterrows():
    answer, _ = solver.solve_problem(row['problem'])
    if answer == row['answer']:
        correct += 1

accuracy = correct / len(reference) * 100
print(f"Accuracy: {accuracy:.2f}%")
```

---

## 💡 Tips for Better Performance

### 1. **More Training Data**
- Generate more simple problems (1000+)
- Use GPT-4 to create step-by-step solutions
- Include diverse problem types

### 2. **Better Base Model**
- Start with math-specialized models:
  - `llemma/llemma-7b` (math-focused)
  - `Qwen/Qwen-Math-7B`
  - `microsoft/phi-3-mini-128k`

### 3. **Two-Stage Training**
- Stage 1: Train on simple problems
- Stage 2: Fine-tune on olympiad problems

### 4. **Ensemble Methods**
- Train multiple models
- Average predictions
- Use voting for discrete answers

### 5. **Chain-of-Thought**
- Format training data with explicit reasoning steps
- Encourage step-by-step solutions

---

## ⚠️ Common Issues

### Out of Memory
```python
# Solution 1: Reduce batch size
batch_size=1

# Solution 2: Use gradient checkpointing
model.gradient_checkpointing_enable()

# Solution 3: Use smaller model
model_name='EleutherAI/pythia-410m'
```

### Slow Training
```python
# Use mixed precision
fp16=True

# Increase batch size with gradient accumulation
per_device_train_batch_size=1
gradient_accumulation_steps=8
```

### Poor Accuracy
```python
# Train longer
num_epochs=10

# More data
create_simple_problems(num_examples=1000)

# Lower learning rate
learning_rate=1e-5
```

---

## 📈 Expected Results

### Simple Problems (test.csv)
- Symbolic solver: ~100% accuracy
- Trained SLM: 90-100% accuracy

### Complex Olympiad Problems (reference.csv)
- Symbolic solver: ~0% accuracy
- Trained SLM (small data): 5-15% accuracy
- Trained SLM (large data + GPT-4 solutions): 30-50% accuracy
- Ensemble + LLM fallback: 50-70% accuracy

---

## 🎓 Next Steps

1. ✅ Run data preparation
2. ✅ Train initial model
3. ⬜ Evaluate on validation set
4. ⬜ Generate more training data (use GPT-4 for solutions)
5. ⬜ Re-train with augmented data
6. ⬜ Test ensemble approaches
7. ⬜ Deploy to Kaggle

---

## 📚 Resources

- **Transformers Docs:** https://huggingface.co/docs/transformers
- **PEFT/LoRA:** https://huggingface.co/docs/peft
- **Training Guide:** https://huggingface.co/docs/transformers/training

---

## 🤝 Contributing

Found improvements? Ideas for better strategies? Feel free to modify and enhance the pipeline!

Good luck with the competition! 🏆
