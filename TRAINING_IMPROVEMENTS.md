# Training Improvements for Curriculum Learning

## Current Issues

### Stage 1: 34% Accuracy
- Model calculates correctly but generates gibberish after answer
- Example: "3 × 2 = 6 Final Answer: 66 6 6 6 = 222..."
- Root cause: No EOS token after "Final Answer: X"

### Stage 2: 88% Accuracy
- Works much better!
- Multi-step reasoning is solid
- Still has some formatting noise

## Recommended Fixes

### 1. Add EOS Tokens to Training Data (BEST FIX)

Modify the training data formatter to add EOS token:

```python
# Current format:
text = f"Problem: {problem}\n\nSolution:\n{steps}\n\nFinal Answer: {answer}"

# Fixed format:
text = f"Problem: {problem}\n\nSolution:\n{steps}\n\nFinal Answer: {answer}{tokenizer.eos_token}"
```

This teaches the model to STOP after the answer.

### 2. Train with Answer-Only Format

For Stage 1 (simple arithmetic), use minimal format:

```python
# Simple format - just the calculation
text = f"Problem: {problem}\n\nSolution:\n{calculation} = {answer}{tokenizer.eos_token}"

# Example:
"Problem: Calculate 3 × 2\n\nSolution:\n3 × 2 = 6<|endoftext|>"
```

### 3. Add Early Stopping During Training Loss Calculation

Mask out tokens after the answer during loss calculation:

```python
# In training loop, find where answer ends and mask subsequent tokens
# This prevents the model from learning to generate after the answer
```

### 4. Use Constrained Generation During Inference

Add stopping criteria during generation:

```python
# Stop when we see "Final Answer: <number>"
from transformers import StoppingCriteria

class AnswerStoppingCriteria(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        # Check if "Final Answer: <number>" pattern is complete
        decoded = tokenizer.decode(input_ids[0])
        if re.search(r'Final Answer:\s*\d+', decoded):
            return True
        return False
```

## Next Steps

1. **Immediate fix**: Update `train_curriculum.py` or notebook to add EOS tokens
2. **Retrain Stage 1** with the fixed format
3. **Optional**: Retrain Stage 2 for even better results
4. **Long-term**: Implement answer masking in training loop

## Implementation Priority

1. ✅ **High Priority**: Add EOS tokens to training format
2. ✅ **Medium Priority**: Simplify Stage 1 format (remove verbose "Final Answer:")
3. ⚠️ **Low Priority**: Custom stopping criteria (can work around with extraction)
4. ⚠️ **Optional**: Token masking (more complex, might not be needed)

## Expected Improvements

With EOS tokens added:
- **Stage 1**: 34% → ~80-90% accuracy
- **Stage 2**: 88% → ~95%+ accuracy

The model already knows HOW to solve the problems - it just needs to learn WHEN to stop!
