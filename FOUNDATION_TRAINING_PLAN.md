# 🎯 Foundation Training Plan - Getting to 20% Accuracy

## 🔍 Problem Diagnosis

**Your current issue:**
- ✅ Model generates text (not completely broken)
- ❌ 0% accuracy with "wrong_answer" errors
- ⚠️ **CRITICAL**: Vocab mismatch (50304 vs 544)

**Root Cause:** Your trained model uses a different tokenizer (50k vocab) than evaluation (544 vocab). This makes the model unable to understand test inputs, guaranteeing 0% accuracy.

---

## 🚀 Solution: Train Fresh Foundation Model

### Option 1: Quick Local Training (2-4 hours, CPU)

**Best for:** Getting started quickly, verifying everything works

```bash
# Activate environment
.\.venv\Scripts\activate

# Train tiny model on simple problems
python train_foundation.py

# This will:
# - Use 256 hidden size, 6 layers (~20M params)
# - Train on 1000 simple MATH problems
# - Save to checkpoints/foundation/
# - Target: 20-30% accuracy
```

**Expected outcome:**
- Training time: 2-4 hours on CPU
- Accuracy: 20-35% on simple problems
- This proves the pipeline works!

### Option 2: Colab Training with Fixed Config (Recommended)

Update your Colab notebook with these critical fixes:

#### Critical Fix #1: Consistent Tokenizer

**Problem:** Your notebook doesn't ensure the same tokenizer is used throughout.

**Fix:** Add this cell BEFORE creating datasets:

```python
# Initialize tokenizer FIRST and reuse it everywhere
from src.tokenizer.math_tokenizer import MathTokenizer

tokenizer = MathTokenizer()
VOCAB_SIZE = len(tokenizer)  # Should be 544

print(f"✅ Tokenizer initialized: {VOCAB_SIZE} tokens")
print(f"   This MUST match the model's vocab_size!")

# Later when creating model config:
config = get_config(MODEL_SIZE)
config.vocab_size = VOCAB_SIZE  # CRITICAL: Set to match tokenizer!
```

#### Critical Fix #2: Start Smaller

**Problem:** 85M params is too large for 7.5k examples (ratio: 0.09 examples/param)

**Fix:** Use "tiny" configuration first:

```python
# In Step 3: Configuration
MODEL_SIZE = "tiny"  # NOT "small"!

# This gives you:
# - ~20M parameters (4x smaller)
# - Better data-to-param ratio (0.4 examples/param)
# - Faster training
# - Less overfitting
```

#### Critical Fix #3: Filter for Easy Problems

```python
# In Step 4: Load dataset
def convert_hf_to_schema(hf_dataset):
    problems = []
    
    for i, item in enumerate(tqdm(hf_dataset)):
        # FILTER: Only Level 1-2 (easy/medium)
        level = item.get('level', 3)
        if level > 2:  # Skip hard problems for foundation
            continue
        
        # ... rest of conversion
```

#### Critical Fix #4: Reduce Sequence Length

```python
# In Step 3: Configuration
MAX_LENGTH = 512  # NOT 1024!

# Reasons:
# - Easy problems have shorter solutions
# - Faster training (2x speedup)
# - Less memory usage
```

#### Complete Fixed Configuration:

```python
# ============================
# FOUNDATION TRAINING CONFIG
# ============================

MODEL_SIZE = "tiny"           # ~20M params
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 5e-4          # Higher LR for smaller model
MAX_EPOCHS = 30
MAX_LENGTH = 512              # Shorter sequences

USE_FULL_DATASET = True
DIFFICULTY_FILTER = [1, 2]    # Easy and Medium only

# This configuration should achieve 20-35% accuracy
```

---

## 📋 Step-by-Step Execution Plan

### Phase 1: Verify Pipeline (30 min)

```bash
# Test that everything works locally
python train_foundation.py

# Wait for first epoch to complete
# Check that:
# - Loss decreases
# - No errors occur
# - Checkpoints are saved
```

### Phase 2: Train Foundation Model

**Local (CPU - 2-4 hours):**
```bash
# Full training run
python train_foundation.py

# Results in: checkpoints/foundation/best_model.pt
```

**Colab (GPU - 1-2 hours):**
```python
# Use fixed notebook (see above)
# Run all cells
# Download best_model.pt when done
```

### Phase 3: Evaluate Foundation

```bash
# Evaluate the trained model
python evaluate_foundation.py --checkpoint checkpoints/foundation/best_model.pt

# This checks:
# ✅ Tokenizer consistency
# ✅ Actual accuracy on test set  
# ✅ Answer extraction rate
```

### Phase 4: Verify Success

**Success criteria:**
- ✅ Accuracy ≥ 20%
- ✅ Answer extraction ≥ 70%
- ✅ No vocab mismatch warnings
- ✅ Loss decreased during training

**If successful:**
→ Ready for fine-tuning! 🎉

**If not successful:**
→ See troubleshooting below

---

## 🔧 Troubleshooting

### Issue: Still getting 0% accuracy

**Check 1: Tokenizer consistency**
```python
# After loading checkpoint
checkpoint = torch.load('checkpoints/foundation/best_model.pt')
print(f"Model vocab: {checkpoint['config'].vocab_size}")

tokenizer = MathTokenizer()
print(f"Tokenizer vocab: {len(tokenizer)}")

# These MUST match!
```

**Check 2: Model is actually trained**
```python
# Check training loss decreased
checkpoint = torch.load('checkpoints/foundation/best_model.pt')
print(f"Final val loss: {checkpoint['val_loss']}")

# Should be < 3.0 for foundation model
# If > 5.0, model didn't learn
```

**Check 3: Test set is appropriate**
```python
# Don't test on super hard problems!
# Foundation model should be tested on Level 1-2 only
```

### Issue: Training loss not decreasing

**Possible causes:**
1. Learning rate too high → Try 1e-4
2. Model too large → Use tiny config
3. Gradient explosion → Check for NaN losses
4. Data quality → Verify problems load correctly

### Issue: Out of memory

```python
# Reduce batch size
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8  # Keep effective batch = 16

# Or reduce sequence length
MAX_LENGTH = 256
```

---

## 📊 Expected Results by Configuration

### Tiny Model (~20M params, 1k problems)
- **Training time:** 2-4 hours (CPU), 30-60 min (GPU)
- **Expected accuracy:** 20-35%
- **Best for:** Quick iteration, pipeline verification
- **When to use:** First training, debugging

### Small Model (~85M params, 5k problems)  
- **Training time:** 8-12 hours (CPU), 2-3 hours (GPU)
- **Expected accuracy:** 30-45%
- **Best for:** Solid foundation before fine-tuning
- **When to use:** After tiny model succeeds

### Medium Model (~350M params, 7.5k problems)
- **Training time:** 24+ hours (CPU), 4-6 hours (GPU)
- **Expected accuracy:** 35-50%
- **Best for:** Production-quality foundation
- **When to use:** Final foundation before competition

---

## 🎯 Recommended Path for You

Given your situation (0% accuracy, need foundation):

### Week 1: Get Foundation Working
1. **Day 1-2:** Run `train_foundation.py` locally (tiny model)
2. **Day 3:** Evaluate, verify 20%+ accuracy
3. **Day 4-5:** Run fixed Colab notebook (small model)
4. **Day 6:** Evaluate Colab model
5. **Day 7:** Pick best checkpoint (should have 25-35% accuracy)

### Week 2: Fine-Tuning
1. Start fine-tuning with the best foundation checkpoint
2. Use supervised learning on solution steps
3. Target: 40-50% accuracy

---

## 🚦 Quick Start Command

```bash
# Right now, run this:
cd C:\Users\alpyaman\Desktop\Projects\AI-Mathematical-Olympiad
.\.venv\Scripts\activate

# Start foundation training
python train_foundation.py

# Let it run for 2-4 hours
# Then evaluate:
python evaluate_foundation.py

# If accuracy >= 20%, you have your foundation! ✅
```

---

## ✅ Success Checklist

Before moving to fine-tuning, verify:

- [ ] Model trains without errors
- [ ] Training loss decreases (should reach <2.5)
- [ ] Validation loss decreases (should reach <3.0)  
- [ ] Accuracy ≥ 20% on test set
- [ ] Answer extraction ≥ 70%
- [ ] No tokenizer mismatch warnings
- [ ] Checkpoints save successfully
- [ ] Can load and run model for inference

**When all checked:** You have a solid foundation! 🎉

---

## 📚 Key Learnings

1. **Tokenizer consistency is CRITICAL** - Same tokenizer for train & eval
2. **Start small** - Tiny model first, scale up later
3. **Filter dataset** - Easy problems for foundation
4. **Verify early** - Check after 1-2 epochs that loss decreases
5. **Monitor metrics** - Not just loss, but actual accuracy

---

## 💡 Pro Tips

- **Save time:** Train tiny model first (2hrs) before committing to larger model (12hrs)
- **Use GPU:** Colab free tier is fine, 6-10x faster than CPU
- **Check often:** Don't wait for full training to realize something is wrong
- **Keep notes:** Track what configurations worked/failed

---

Need help? Check if:
1. Tokenizer matches (should be 544 tokens)
2. Model is tiny config first
3. Training on Level 1-2 problems only
4. Loss is actually decreasing

Good luck! 🚀
