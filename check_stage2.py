"""
Check Stage 2 Checkpoint - Multi-step arithmetic

This script evaluates the stage2_complete.pt checkpoint.
"""

import torch
import random
import sys
from pathlib import Path
import re

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.model.decoder import MathTransformerDecoder
from src.tokenizer.math_tokenizer import MathTokenizer
from src.config.model_config import get_small_config
from src.data.data_schema import (
    MathProblem, MathSolution, ReasoningStep, 
    DifficultyLevel, ProblemType
)
from src.evaluation.answer_extraction import compare_answers


def create_stage2_multistep(num_problems=50):
    """Stage 2: Two-step arithmetic"""
    problems = []
    random.seed(43)
    
    for i in range(num_problems):
        choice = random.choice(['add_sub', 'mul_add'])
        
        if choice == 'add_sub':
            a, b, c = random.randint(10, 30), random.randint(5, 20), random.randint(1, 10)
            answer = a + b - c
            problem_text = f"Calculate {a} + {b} - {c}"
            steps_text = f"First: {a} + {b} = {a+b}, then {a+b} - {c} = {answer}"
        else:  # mul_add
            a, b, c = random.randint(2, 10), random.randint(2, 10), random.randint(1, 20)
            answer = a * b + c
            problem_text = f"Calculate {a} × {b} + {c}"
            steps_text = f"First: {a} × {b} = {a*b}, then {a*b} + {c} = {answer}"
        
        problems.append(MathProblem(
            problem_id=f"s2_{i}",
            problem_statement=problem_text,
            solution=MathSolution(
                steps=[ReasoningStep(1, "Multistep", steps_text, None)],
                final_answer=str(answer),
                answer_type="integer"
            ),
            difficulty=DifficultyLevel.MEDIUM,
            problem_type=ProblemType.ALGEBRA,
            topics=["arithmetic"],
            source="curriculum_stage2"
        ))
    
    return problems


def extract_stage2_answer(generated_text):
    """Extract answer for Stage 2 - look for answer after 'Final Answer:' or last '= number' before it."""
    # Strategy 1: Look for "Final Answer: number" pattern
    final_answer_pattern = r'Final Answer:\s*(-?\d+(?:\.\d+)?)'
    match = re.search(final_answer_pattern, generated_text)
    if match:
        return match.group(1)
    
    # Strategy 2: If "Final Answer:" exists but no number after it, 
    # get the last '= number' BEFORE "Final Answer:"
    if "Final Answer:" in generated_text:
        idx = generated_text.index("Final Answer:")
        text_before = generated_text[:idx]
        equals_pattern = r'=\s*(-?\d+(?:\.\d+)?)'  
        matches = re.findall(equals_pattern, text_before)
        if matches:
            return matches[-1]
    
    # Strategy 3: No "Final Answer:" found, look for last '= number' in entire text
    equals_pattern = r'=\s*(-?\d+(?:\.\d+)?)'  
    matches = re.findall(equals_pattern, generated_text)
    if matches:
        return matches[-1]
    
    return None


def load_checkpoint(checkpoint_path: str, device='cpu'):
    """Load model from checkpoint."""
    print(f"\n📦 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("   ✅ Using config from checkpoint")
    else:
        config = get_small_config()
        print("   ⚠️ Using default small config")
    
    model = MathTransformerDecoder(config)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    tokenizer = MathTokenizer()
    
    print(f"   Layers: {config.num_hidden_layers}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Vocab size: {config.vocab_size}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if 'epoch' in checkpoint:
        print(f"   Trained epochs: {checkpoint['epoch']}")
    if 'train_loss' in checkpoint:
        print(f"   Training loss: {checkpoint['train_loss']:.4f}")
    
    return model, tokenizer, config


def evaluate_model(model, tokenizer, problems, device='cpu'):
    """Evaluate model on problems."""
    model.eval()
    model.to(device)
    
    correct = 0
    extracted = 0
    failed = 0
    
    print(f"\n🧪 Evaluating on {len(problems)} problems...")
    
    with torch.no_grad():
        for idx, prob in enumerate(problems):
            if idx > 0 and idx % 10 == 0:
                print(f"   Progress: {idx}/{len(problems)} - Accuracy so far: {(correct/idx)*100:.1f}%")
            
            try:
                prompt = f"Problem: {prob.problem_statement}\n\nSolution:"
                encoded = tokenizer.encode(prompt, add_special_tokens=False, max_length=256, truncation=True)
                input_ids = torch.tensor([encoded['input_ids']]).to(device)
                
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=100,  # Stage 2 needs more tokens for multi-step
                    temperature=0.0,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                generated = tokenizer.decode(
                    outputs[0][input_ids.shape[1]:].tolist(),
                    skip_special_tokens=True
                )
                
                # Extract using Stage 2 logic (last = sign)
                predicted = extract_stage2_answer(generated)
                ground_truth = prob.solution.final_answer
                
                if predicted:
                    extracted += 1
                    if compare_answers(predicted, ground_truth, tolerance=1e-4):
                        correct += 1
                        if idx < 5:
                            print(f"   ✅ {prob.problem_statement}")
                            print(f"      Generated: {generated[:100]}...")
                            print(f"      Extracted: {predicted} ✓")
                    else:
                        if idx < 5:
                            print(f"   ❌ {prob.problem_statement}")
                            print(f"      Generated: {generated[:100]}...")
                            print(f"      Extracted: {predicted}, Expected: {ground_truth}")
                else:
                    if idx < 5:
                        print(f"   ⚠️ {prob.problem_statement}")
                        print(f"      Generated: {generated[:100]}...")
                        print("      Could not extract answer")
                        
            except Exception as e:
                failed += 1
                if idx < 5:
                    print(f"   💥 Error: {e}")
                continue
    
    total = len(problems)
    accuracy = (correct / total) * 100 if total > 0 else 0
    extraction_rate = (extracted / total) * 100 if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'extraction_rate': extraction_rate,
        'correct': correct,
        'extracted': extracted,
        'failed': failed,
        'total': total
    }


def main():
    print("="*70)
    print("STAGE 2 CHECKPOINT EVALUATION")
    print("Multi-step Arithmetic (e.g., 3 × 9 + 20 = 47)")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️ Using device: {device}")
    
    # Create test dataset
    print("\n📝 Creating test dataset...")
    problems = create_stage2_multistep(num_problems=100)
    print(f"   Total problems: {len(problems)}")
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/curriculum/stage2_complete.pt'
    model, tokenizer, config = load_checkpoint(checkpoint_path, device)
    
    # Evaluate
    results = evaluate_model(model, tokenizer, problems, device)
    
    # Print results
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("="*70)
    print(f"Accuracy:        {results['accuracy']:.2f}%")
    print(f"Extraction Rate: {results['extraction_rate']:.2f}%")
    print(f"Correct:         {results['correct']}/{results['total']}")
    print(f"Extracted:       {results['extracted']}/{results['total']}")
    print(f"Failed:          {results['failed']}/{results['total']}")
    
    if results['accuracy'] >= 80:
        print("\n✅ Stage 2 checkpoint is performing well!")
    elif results['accuracy'] >= 50:
        print("\n⚠️ Stage 2 checkpoint needs improvement")
    else:
        print("\n❌ Stage 2 checkpoint has significant issues")


if __name__ == '__main__':
    main()
