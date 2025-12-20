"""
Foundation Model Evaluation
Quick evaluation to verify the model is learning and achieving 20%+ accuracy
"""

import torch
from pathlib import Path
from tqdm import tqdm
import json

from src.model.decoder import MathTransformerDecoder
from src.tokenizer.math_tokenizer import MathTokenizer
from src.evaluation.answer_extraction import AnswerExtractor, compare_answers

def evaluate_foundation(checkpoint_path: str):
    """Evaluate foundation model"""
    print("\n" + "="*70)
    print("📊 FOUNDATION MODEL EVALUATION")
    print("="*70 + "\n")
    
    # Load checkpoint
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Initialize tokenizer (MUST use same one as training!)
    tokenizer = MathTokenizer()
    print(f"   Tokenizer vocab size: {len(tokenizer)}")
    
    # Get config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("   Model config from checkpoint:")
        print(f"     - Layers: {config.num_hidden_layers}")
        print(f"     - Hidden size: {config.hidden_size}")
        print(f"     - Vocab size: {config.vocab_size}")
        
        # CRITICAL CHECK
        if config.vocab_size != len(tokenizer):
            print("\n⚠️  WARNING: Vocab size mismatch!")
            print(f"   Checkpoint: {config.vocab_size}")
            print(f"   Current tokenizer: {len(tokenizer)}")
            print("   This will cause 0% accuracy!")
            return
    else:
        print("   ⚠️  No config in checkpoint, evaluation may fail")
        return
    
    # Create model
    model = MathTransformerDecoder(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"   Device: {device}")
    print("✅ Model loaded successfully\n")
    
    # Load test problems
    print("📚 Loading test problems...")
    problems = []
    
    # Try Method 1: Load from HuggingFace
    try:
        from datasets import load_dataset
        print("   Attempting to load MATH dataset from HuggingFace...")
        dataset = load_dataset("lighteval/MATH", split="test")
        
        from src.data.data_schema import MathProblem, MathSolution, ReasoningStep, DifficultyLevel, ProblemType
        import re
        
        # Convert to our schema (only Level 1-2)
        for i, item in enumerate(dataset):
            level = item.get('level', 3)
            if level > 2:  # Only easy/medium
                continue
            
            if len(problems) >= 100:  # Limit to 100
                break
            
            # Extract answer
            final_answer = item.get('solution', 'Unknown')
            matches = re.findall(r'\\boxed{(.+?)}', str(final_answer))
            if matches:
                final_answer = matches[-1]
            
            sol = MathSolution(
                steps=[ReasoningStep(1, "Solution", item.get('solution', ''), None)],
                final_answer=final_answer,
                answer_type="exact"
            )
            
            prob = MathProblem(
                problem_id=f"test_{i}",
                problem_statement=item['problem'],
                solution=sol,
                difficulty=DifficultyLevel.EASY if level == 1 else DifficultyLevel.MEDIUM,
                problem_type=ProblemType.ALGEBRA,
                topics=[item.get('type', 'math')],
                source="MATH"
            )
            problems.append(prob)
        
        print(f"   ✅ Loaded {len(problems)} problems from HuggingFace MATH dataset")
    except Exception as e:
        print(f"   ⚠️  Could not load from HuggingFace: {e}")
    
    # Try Method 2: Create synthetic test problems
    if len(problems) == 0:
        print("   Creating synthetic test problems...")
        from src.data.data_schema import MathProblem, MathSolution, ReasoningStep, DifficultyLevel, ProblemType
        import random
        
        # Simple arithmetic problems to test if model learned anything
        for i in range(50):
            a, b = random.randint(1, 20), random.randint(1, 20)
            answer = a + b
            problems.append(MathProblem(
                problem_id=f"test_add_{i}",
                problem_statement=f"What is {a} + {b}?",
                solution=MathSolution(
                    steps=[ReasoningStep(1, "Add", f"{a} + {b} = {answer}", None)],
                    final_answer=str(answer),
                    answer_type="integer"
                ),
                difficulty=DifficultyLevel.EASY,
                problem_type=ProblemType.ALGEBRA,
                topics=["arithmetic"],
                source="synthetic"
            ))
        
        for i in range(50):
            a, b = random.randint(2, 12), random.randint(2, 12)
            answer = a * b
            problems.append(MathProblem(
                problem_id=f"test_mul_{i}",
                problem_statement=f"Calculate {a} × {b}",
                solution=MathSolution(
                    steps=[ReasoningStep(1, "Multiply", f"{a} × {b} = {answer}", None)],
                    final_answer=str(answer),
                    answer_type="integer"
                ),
                difficulty=DifficultyLevel.EASY,
                problem_type=ProblemType.ALGEBRA,
                topics=["arithmetic"],
                source="synthetic"
            ))
        
        print(f"   ✅ Created {len(problems)} synthetic test problems")
    
    if len(problems) == 0:
        print("   ❌ No test data available")
        return
    
    # Limit to reasonable size
    if len(problems) > 100:
        problems = problems[:100]
    
    print(f"   Evaluating on {len(problems)} problems\n")
    
    # Evaluation
    extractor = AnswerExtractor()
    results = []
    correct = 0
    extracted = 0
    
    print("🔬 Evaluating...")
    for problem in tqdm(problems):
        # Create prompt
        prompt = f"Problem: {problem.problem_statement}\n\nSolution:"
        
        # Encode
        try:
            encoded = tokenizer.encode(prompt, max_length=512, truncation=True)
            input_ids = torch.tensor([encoded['input_ids']]).to(device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=150,
                    temperature=0.0,  # Greedy for consistency
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode
            generated = tokenizer.decode(
                outputs[0][input_ids.shape[1]:].tolist(),
                skip_special_tokens=True
            )
            
            # Extract answer
            predicted = extractor.extract(generated)
            ground_truth = problem.solution.final_answer
            
            is_correct = False
            if predicted:
                extracted += 1
                is_correct = compare_answers(predicted, ground_truth, tolerance=1e-4)
                if is_correct:
                    correct += 1
            
            results.append({
                'problem_id': problem.problem_id,
                'problem': problem.problem_statement,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'is_correct': is_correct,
                'generated': generated[:200]  # Store first 200 chars
            })
            
        except Exception as e:
            print(f"\n   Error on {problem.problem_id}: {e}")
            results.append({
                'problem_id': problem.problem_id,
                'problem': problem.problem_statement,
                'ground_truth': problem.solution.final_answer,
                'predicted': None,
                'is_correct': False,
                'generated': f"Error: {e}"
            })
    
    # Results
    total = len(problems)
    accuracy = (correct / total * 100) if total > 0 else 0
    extraction_rate = (extracted / total * 100) if total > 0 else 0
    
    print(f"\n{'='*70}")
    print("📊 RESULTS")
    print(f"{'='*70}")
    print(f"Total Problems:        {total}")
    print(f"Correct:               {correct}")
    print(f"Accuracy:              {accuracy:.2f}%")
    print(f"Answer Extraction:     {extraction_rate:.2f}%")
    print(f"{'='*70}\n")
    
    # Analysis
    if accuracy >= 20:
        print("✅ SUCCESS! Model achieved 20%+ accuracy")
        print("   Ready for fine-tuning")
    elif accuracy >= 10:
        print("⚠️  MODERATE: 10-20% accuracy")
        print("   Consider training longer or with more data")
    elif accuracy > 0:
        print("⚠️  LOW: <10% accuracy")
        print("   Model is learning but needs more training")
    else:
        print("❌ FAILURE: 0% accuracy")
        print("   Check tokenizer consistency and training")
    
    # Show examples
    print(f"\n{'='*70}")
    print("📝 SAMPLE RESULTS")
    print(f"{'='*70}\n")
    
    # Show 3 correct
    correct_examples = [r for r in results if r['is_correct']]
    if correct_examples:
        print("✅ Correct Examples:\n")
        for i, ex in enumerate(correct_examples[:3], 1):
            print(f"{i}. Problem: {ex['problem'][:60]}...")
            print(f"   Answer: {ex['ground_truth']}")
            print(f"   Predicted: {ex['predicted']}")
            print()
    
    # Show 3 incorrect
    incorrect_examples = [r for r in results if not r['is_correct']]
    if incorrect_examples:
        print("❌ Incorrect Examples:\n")
        for i, ex in enumerate(incorrect_examples[:3], 1):
            print(f"{i}. Problem: {ex['problem'][:60]}...")
            print(f"   Answer: {ex['ground_truth']}")
            print(f"   Predicted: {ex['predicted']}")
            print(f"   Generated: {ex['generated'][:100]}...")
            print()
    
    # Save results
    checkpoint_dir = Path(checkpoint_path).parent
    results_file = checkpoint_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'extraction_rate': extraction_rate,
            'total': total,
            'correct': correct,
            'results': results
        }, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")
    print(f"{'='*70}\n")
    
    return accuracy

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase2/best_model.pt")
    args = parser.parse_args()
    
    evaluate_foundation(args.checkpoint)
