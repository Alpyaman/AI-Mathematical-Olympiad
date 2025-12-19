# 📊 Evaluation Guide

Complete guide for evaluating your mathematical reasoning model.

---

## 🎯 Quick Start

### 1. Evaluate on Test Set

```bash
# Activate environment
.\.venv\Scripts\activate

# Run evaluation on test set
python evaluate_model.py \
    --checkpoint checkpoints/pretraining_notebook/final.pt \
    --config small \
    --mode evaluate \
    --test-data data/test.csv \
    --output-dir evaluation_results \
    --batch-size 8
```

### 2. Generate Kaggle Submission

```bash
# Single prediction per problem
python evaluate_model.py \
    --checkpoint checkpoints/pretraining_notebook/final.pt \
    --config small \
    --mode kaggle \
    --test-csv data/test.csv \
    --submission-path submission.csv \
    --batch-size 8
```

### 3. Ensemble Submission (Better Accuracy)

```bash
# Multiple attempts per problem (takes 3x longer)
python evaluate_model.py \
    --checkpoint checkpoints/pretraining_notebook/final.pt \
    --config small \
    --mode kaggle \
    --test-csv data/test.csv \
    --submission-path submission_ensemble.csv \
    --ensemble \
    --num-attempts 3 \
    --batch-size 4
```

---

## 📈 What You Get

### Evaluation Mode Outputs

```
evaluation_results/
├── evaluation_results.json       # Complete metrics in JSON
├── detailed_results.csv          # Per-problem results
└── evaluation_summary.txt        # Human-readable report
```

### Metrics Provided

**Overall Metrics:**
- ✅ Accuracy (% correct answers)
- ✅ Answer extraction rate (% answers found)
- ✅ Total correct/incorrect

**Breakdown Metrics:**
- 📊 By difficulty level (Easy/Medium/Hard/Olympiad)
- 📚 By topic (Algebra/Geometry/Number Theory/etc.)
- ⚠️ Error analysis (types of failures)

---

## 🎓 Understanding the Results

### Example Output

```
📊 EVALUATION SUMMARY
================================================================================

✅ Overall Accuracy: 42.50%
   Correct: 85/200
   Answer Extraction Rate: 92.00%

📈 Performance by Difficulty:
   EASY           : 75.00% (30/40)
   MEDIUM         : 50.00% (40/80)
   HARD           : 25.00% (15/60)
   OLYMPIAD       : 0.00% (0/20)

📚 Top Topics by Accuracy:
   algebra        : 55.00% (44/80)
   geometry       : 40.00% (20/50)
   number_theory  : 30.00% (15/50)

⚠️ Error Analysis:
   Total Errors: 115
   wrong_answer: 95
   no_answer_extracted: 20
```

### What This Tells You

1. **Overall Accuracy 42.5%** - Your model gets about 4/10 problems correct
   - For reference: Random guessing ~5-10%, human expert ~90%+
   - First training run 20-50% is normal!

2. **Answer Extraction 92%** - Model formats answers correctly
   - High rate (>80%) is good - means model learned output format
   - Low rate (<50%) means model struggles with answer formatting

3. **Performance by Difficulty** - Shows where model excels/struggles
   - Easy problems: Should be >60%
   - Medium: Target 30-50%
   - Hard: Target 10-30%
   - Olympiad: <10% is normal for initial models

4. **Performance by Topic** - Which math areas are strongest
   - Use this to focus training data
   - Weak topics need more examples

---

## 🎯 How to Improve Results

### If Accuracy is Low (<20%)

1. **Check Training** - Did model train long enough?
```bash
# Look at training logs
# Check loss decreased over time
# Verify model actually learned something
```

2. **Check Answer Format** - Is extraction rate low?
```python
# Review some generated solutions manually
# Check if model produces \\boxed{answer} format
```

3. **Simplify First** - Test on easier problems
```bash
# Filter for EASY difficulty only
# Make sure model can solve simple problems first
```

### If Accuracy is Medium (20-40%)

✅ **This is GOOD for first training!**

1. **Train Longer** - More epochs/more data
2. **Fine-tune** - Use supervised fine-tuning on solutions
3. **Adjust Generation** - Try different temperature/top-p

### If Accuracy is Good (>40%)

🎉 **Excellent progress!**

1. **Scale Up** - Try base config (1B params)
2. **Ensemble** - Use multiple attempts
3. **Submit to Kaggle** - See how you rank!

---

## 🔍 Debugging Common Issues

### Issue: "No answers extracted"

**Problem:** Model generates text but no \\boxed{} answers

**Solutions:**
```python
# 1. Check if training data had boxed answers
# 2. Add answer formatting to prompts
# 3. Fine-tune specifically on answer extraction
```

### Issue: "All predictions are wrong"

**Problem:** Model generates answers but all incorrect

**Solutions:**
```python
# 1. Verify model checkpoint loaded correctly
# 2. Check if model is just memorizing
# 3. Evaluate on training set first (should be higher)
# 4. Review actual generated solutions manually
```

### Issue: "OOM (Out of Memory)"

**Problem:** GPU runs out of memory during evaluation

**Solutions:**
```bash
# Reduce batch size
--batch-size 1

# Reduce max length
--max-length 256

# Use CPU (slower)
--device cpu
```

---

## 📝 Manual Inspection

Always look at actual examples!

```python
# Load detailed results
import pandas as pd
df = pd.read_csv('evaluation_results/detailed_results.csv')

# Look at correct answers
correct = df[df['is_correct'] == True].head(5)
print(correct[['problem', 'generated', 'extracted_answer']])

# Look at mistakes
incorrect = df[df['is_correct'] == False].head(5)
print(incorrect[['problem', 'generated', 'extracted_answer', 'ground_truth']])
```

---

## 🏆 Kaggle Submission Tips

### Submission Format

Your `submission.csv` should look like:
```csv
id,answer
1,42
2,3.14159
3,17
```

### Before Submitting

1. ✅ Check file format is correct
```python
import pandas as pd
sub = pd.read_csv('submission.csv')
print(sub.head())
print(f"Shape: {sub.shape}")
print(f"Columns: {sub.columns.tolist()}")
```

2. ✅ Verify no missing answers
```python
print(f"Missing: {sub['answer'].isna().sum()}")
```

3. ✅ Check for duplicates
```python
print(f"Duplicates: {sub['id'].duplicated().sum()}")
```

### Submission Strategies

**Strategy 1: Single Best Model**
- Train one good model
- Use greedy decoding (temp=0.0)
- Fast, simple

**Strategy 2: Ensemble**
- Generate 3-5 solutions per problem
- Take most common answer
- Slower but more accurate

**Strategy 3: Multiple Models**
- Train different model sizes
- Average or vote on answers
- Best accuracy, most compute

---

## 📊 Tracking Progress

Keep a log of your experiments:

```markdown
| Date | Checkpoint | Config | Accuracy | Notes |
|------|-----------|--------|----------|-------|
| 12/19 | baseline | small | 15.2% | Initial training |
| 12/20 | epoch_50 | small | 32.4% | Trained longer |
| 12/21 | finetuned | small | 41.8% | SFT on solutions |
| 12/22 | ensemble | small | 45.2% | 3x ensemble |
```

---

## 🎯 Next Steps

1. **Run Your First Evaluation**
```bash
python evaluate_model.py --checkpoint checkpoints/pretraining_notebook/final.pt --config small --mode evaluate --test-data data/test.csv
```

2. **Review Results** - Understand strengths/weaknesses

3. **Iterate** - Train more, try different configs

4. **Submit to Kaggle** - Get real competition score!

---

## 💡 Pro Tips

- Start with small test set (50-100 problems) for fast iteration
- Always check a few examples manually
- Compare multiple checkpoints to pick the best
- Use ensemble for final submission
- Track what works in a spreadsheet

---

Good luck! 🚀
