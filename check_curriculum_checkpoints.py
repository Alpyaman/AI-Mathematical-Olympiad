"""
Check Curriculum Training Checkpoints

This script evaluates the stage1 and stage2 checkpoints from curriculum training.
"""

import torch
import random
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.model.decoder import MathTransformerDecoder
from src.tokenizer.math_tokenizer import MathTokenizer
from src.config.model_config import get_small_config
from src.data.data_schema import (
    MathProblem, MathSolution, ReasoningStep, 
    DifficultyLevel, ProblemType
)
from src.evaluation.answer_extraction import AnswerExtractor, compare_answers
import re


def create_stage1_basic(num_problems=100):
    """Stage 1: Single arithmetic operations"""
    problems = []
    random.seed(42)
    
    for i in range(num_problems):
        choice = random.choice(['add', 'sub', 'mul'])
        
        if choice == 'add':
            a, b = random.randint(1, 20), random.randint(1, 20)
            answer = a + b
            problem_text = f"Calculate {a} + {b}"
            steps_text = f"{a} + {b} = {answer}"
        elif choice == 'sub':
            a, b = random.randint(10, 30), random.randint(1, 15)
            answer = a - b
            problem_text = f"Calculate {a} - {b}"
            steps_text = f"{a} - {b} = {answer}"
        else:  # mul
            a, b = random.randint(2, 10), random.randint(2, 10)
            answer = a * b
            problem_text = f"Calculate {a} × {b}"
            steps_text = f"{a} × {b} = {answer}"
        
        problems.append(MathProblem(
            problem_id=f"s1_{i}",
            problem_statement=problem_text,
            solution=MathSolution(
                steps=[ReasoningStep(1, "Compute", steps_text, None)],
                final_answer=str(answer),
                answer_type="integer"
            ),
            difficulty=DifficultyLevel.EASY,
            problem_type=ProblemType.ALGEBRA,
            topics=["arithmetic"],
            source="curriculum_stage1"
        ))
    
    return problems


def create_stage2_multistep(num_problems=100):
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


def load_checkpoint(checkpoint_path: str, device='cpu'):
    """Load model from checkpoint."""
    print(f"\n📦 Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("   ✅ Using config from checkpoint")
    else:
        config = get_small_config()
        print("   ⚠️ Using default small config")
    
    # Initialize model
    model = MathTransformerDecoder(config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Initialize tokenizer
    tokenizer = MathTokenizer()
    
    # Print info
    print(f"   Layers: {config.num_hidden_layers}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Vocab size: {config.vocab_size}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if 'epoch' in checkpoint:
        print(f"   Trained epochs: {checkpoint['epoch']}")
    if 'train_loss' in checkpoint:
        print(f"   Training loss: {checkpoint['train_loss']:.4f}")
    
    return model, tokenizer, config


def extract_answer_improved(generated_text, problem_statement):
    """Extract answer from generated text with improved logic."""
    # Strategy 1: Look for "Final Answer:" followed by a number (highest priority)
    final_answer_pattern = r'Final Answer:\s*(-?\d+(?:\.\d+)?)'
    match = re.search(final_answer_pattern, generated_text)
    if match:
        return match.group(1)
    
    # Strategy 2: For multi-step, look for the LAST "= number" pattern
    # This captures the final answer in "First: X, then Y = answer" format
    equals_pattern = r'=\s*(-?\d+(?:\.\d+)?)'  
    matches = re.findall(equals_pattern, generated_text)
    if matches:
        # Return the LAST number after '=' (final answer in multi-step)
        return matches[-1]
    
    # Strategy 3: Use the standard extractor as fallback
    extractor = AnswerExtractor()
    return extractor.extract(generated_text)


def evaluate_model(model, tokenizer, problems, device='cpu', num_samples=None, verbose=True):
    """Evaluate model on problems."""
    model.eval()
    model.to(device)
    
    extractor = AnswerExtractor()
    
    if num_samples and num_samples < len(problems):
        test_problems = random.sample(problems, num_samples)
    else:
        test_problems = problems
    
    correct = 0
    extracted = 0
    failed = 0
    
    if verbose:
        print(f"\n🧪 Evaluating on {len(test_problems)} problems...")
    
    with torch.no_grad():
        for idx, prob in enumerate(test_problems):
            # Show progress every 10 problems
            if verbose and idx > 0 and idx % 10 == 0:
                print(f"   Progress: {idx}/{len(test_problems)} problems evaluated...")
            try:
                # Use simple format matching training
                prompt = f"Problem: {prob.problem_statement}\n\nSolution:"
                encoded = tokenizer.encode(prompt, add_special_tokens=False, max_length=256, truncation=True)
                input_ids = torch.tensor([encoded['input_ids']]).to(device)
                
                # Generate with early stopping
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=50,  # Reduced from 100 - answers appear early
                    temperature=0.0,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Decode
                generated = tokenizer.decode(
                    outputs[0][input_ids.shape[1]:].tolist(),
                    skip_special_tokens=True
                )
                
                # Truncate after "Final Answer: X" to avoid gibberish
                if "Final Answer:" in generated:
                    # Find the answer after "Final Answer:"
                    final_idx = generated.index("Final Answer:")
                    # Take up to 50 chars after "Final Answer:" to capture the number
                    truncate_point = final_idx + 50
                    remainder = generated[final_idx:truncate_point]
                    # Find first number after "Final Answer:"
                    match = re.search(r'Final Answer:\s*(-?\d+(?:\.\d+)?)', remainder)
                    if match:
                        # Truncate right after the number
                        generated = generated[:final_idx + match.end()]
                
                # Extract answer using improved extraction
                predicted = extract_answer_improved(generated, prob.problem_statement)
                ground_truth = prob.solution.final_answer
                
                if predicted:
                    extracted += 1
                    if compare_answers(predicted, ground_truth, tolerance=1e-4):
                        correct += 1
                        if verbose and idx < 5:
                            print(f"   ✅ Problem: {prob.problem_statement}")
                            print(f"      Generated: {generated[:100]}...")
                            print(f"      Predicted: {predicted}, Expected: {ground_truth}")
                    else:
                        if verbose and idx < 5:
                            print(f"   ❌ Problem: {prob.problem_statement}")
                            print(f"      Generated: {generated[:100]}...")
                            print(f"      Predicted: {predicted}, Expected: {ground_truth}")
                else:
                    if verbose and idx < 5:
                        print(f"   ⚠️ Problem: {prob.problem_statement}")
                        print(f"      Generated: {generated[:100]}...")
                        print("      Could not extract answer")
                        
            except Exception as e:
                failed += 1
                if verbose and idx < 5:
                    print(f"   💥 Error: {e}")
                continue
    
    total = len(test_problems)
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
    print("CURRICULUM CHECKPOINT EVALUATION")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️ Using device: {device}")
    
    # Create test datasets
    print("\n📝 Creating test datasets...")
    stage1_problems = create_stage1_basic(num_problems=50)  # Reduced for faster testing
    stage2_problems = create_stage2_multistep(num_problems=50)  # Reduced for faster testing
    print(f"   Stage 1: {len(stage1_problems)} problems")
    print(f"   Stage 2: {len(stage2_problems)} problems")
    
    # Define checkpoints
    checkpoints = [
        ('checkpoints/curriculum/stage1_complete.pt', 'Stage 1', stage1_problems),
        ('checkpoints/curriculum/stage2_complete.pt', 'Stage 2', stage2_problems),
    ]
    
    results_summary = []
    
    for checkpoint_path, stage_name, test_problems in checkpoints:
        print("\n" + "="*70)
        print(f"EVALUATING {stage_name.upper()}")
        print("="*70)
        
        try:
            # Load checkpoint
            model, tokenizer, config = load_checkpoint(checkpoint_path, device)
            
            # Evaluate on corresponding stage problems
            print(f"\nTesting on {stage_name} problems:")
            results = evaluate_model(model, tokenizer, test_problems, device, verbose=True)
            
            print("\n📊 RESULTS:")
            print(f"   Accuracy: {results['accuracy']:.2f}%")
            print(f"   Extraction Rate: {results['extraction_rate']:.2f}%")
            print(f"   Correct: {results['correct']}/{results['total']}")
            print(f"   Extracted: {results['extracted']}/{results['total']}")
            print(f"   Failed: {results['failed']}/{results['total']}")
            
            results_summary.append({
                'stage': stage_name,
                'checkpoint': checkpoint_path,
                'accuracy': results['accuracy'],
                'extraction_rate': results['extraction_rate']
            })
            
            # Also test on stage 1 problems to see if it learned basics
            if stage_name == 'Stage 2':
                print("\nTesting Stage 2 checkpoint on Stage 1 problems:")
                results_s1 = evaluate_model(model, tokenizer, stage1_problems, device, num_samples=50, verbose=False)
                print(f"   Accuracy on Stage 1: {results_s1['accuracy']:.2f}%")
                print(f"   Extraction Rate: {results_s1['extraction_rate']:.2f}%")
            
        except Exception as e:
            print(f"\n❌ Error evaluating {stage_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for result in results_summary:
        print(f"\n{result['stage']}:")
        print(f"   Checkpoint: {result['checkpoint']}")
        print(f"   Accuracy: {result['accuracy']:.2f}%")
        print(f"   Extraction Rate: {result['extraction_rate']:.2f}%")


if __name__ == '__main__':
    main()
