"""
Comprehensive Evaluation on Large Test Set
Tests model on 100+ problems to get reliable accuracy estimate
"""

import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import argparse

from src.model.decoder import MathTransformerDecoder
from src.tokenizer.math_tokenizer import MathTokenizer
from src.evaluation.answer_extraction import AnswerExtractor, compare_answers

def load_reference_problems():
    """Load the 10 reference problems"""
    df = pd.read_csv('data/reference.csv')
    problems = []
    
    for _, row in df.iterrows():
        problems.append({
            'id': row['id'],
            'problem': row['problem'],
            'answer': str(row['answer'])
        })
    
    return problems

def load_math_dataset_hf(num_problems=100):
    """Try to load MATH dataset from HuggingFace"""
    try:
        from datasets import load_dataset
        print("   Loading MATH dataset from HuggingFace...")
        
        # Try different sources
        datasets_to_try = [
            ("hendrycks/math", "train"),
            ("lighteval/MATH", "train"),
            ("competition_math", "train"),
        ]
        
        dataset = None
        for ds_name, split in datasets_to_try:
            try:
                dataset = load_dataset(ds_name, split=split)
                print(f"   ✅ Loaded from {ds_name}")
                break
            except Exception:
                continue
        
        if dataset is None:
            return []
        
        import re
        problems = []
        
        for i, item in enumerate(dataset):
            if len(problems) >= num_problems:
                break
            
            # Extract answer from solution
            solution_text = str(item.get('solution', ''))
            matches = re.findall(r'\\boxed\{([^}]+)\}', solution_text)
            
            if matches:
                answer = matches[-1]
                problems.append({
                    'id': f'math_{i}',
                    'problem': item['problem'],
                    'answer': answer
                })
        
        return problems
        
    except Exception as e:
        print(f"   ⚠️  Could not load from HuggingFace: {e}")
        return []

def create_synthetic_problems(num_problems=200):
    """Create synthetic math problems for testing"""
    import random
    problems = []
    
    # Addition (50 problems)
    for i in range(50):
        a, b = random.randint(1, 100), random.randint(1, 100)
        problems.append({
            'id': f'add_{i}',
            'problem': f'What is {a} + {b}?',
            'answer': str(a + b)
        })
    
    # Subtraction (50 problems)
    for i in range(50):
        a, b = random.randint(50, 200), random.randint(1, 49)
        problems.append({
            'id': f'sub_{i}',
            'problem': f'Calculate {a} - {b}',
            'answer': str(a - b)
        })
    
    # Multiplication (50 problems)
    for i in range(50):
        a, b = random.randint(2, 20), random.randint(2, 20)
        problems.append({
            'id': f'mul_{i}',
            'problem': f'What is {a} × {b}?',
            'answer': str(a * b)
        })
    
    # Simple equations (50 problems)
    for i in range(50):
        a, b, c = random.randint(2, 10), random.randint(1, 20), random.randint(10, 100)
        x = (c - b) // a
        if a * x + b == c:  # Only include if it has integer solution
            problems.append({
                'id': f'eq_{i}',
                'problem': f'Solve for x: {a}x + {b} = {c}',
                'answer': str(x)
            })
    
    return problems[:num_problems]

def evaluate_model(checkpoint_path, num_problems=100):
    """Run comprehensive evaluation"""
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE MODEL EVALUATION")
    print("="*70 + "\n")
    
    # Load model
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    tokenizer = MathTokenizer()
    config = checkpoint['config']
    
    print(f"   Model: {config.num_hidden_layers} layers, {config.hidden_size} hidden")
    
    model = MathTransformerDecoder(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"   Device: {device}\n")
    
    # Load test problems
    print("📚 Loading test problems...")
    
    # 1. Reference problems (10)
    ref_problems = load_reference_problems()
    print(f"   ✅ {len(ref_problems)} reference problems")
    
    # 2. Try MATH dataset
    math_problems = load_math_dataset_hf(num_problems=num_problems)
    if math_problems:
        print(f"   ✅ {len(math_problems)} MATH dataset problems")
    
    # 3. Synthetic problems
    synthetic_problems = create_synthetic_problems(num_problems=num_problems)
    print(f"   ✅ {len(synthetic_problems)} synthetic problems")
    
    # Combine all
    all_problems = ref_problems + math_problems + synthetic_problems
    print(f"\n   📊 Total: {len(all_problems)} problems\n")
    
    # Evaluate
    extractor = AnswerExtractor()
    results = {
        'reference': {'correct': 0, 'total': 0, 'extracted': 0},
        'math': {'correct': 0, 'total': 0, 'extracted': 0},
        'synthetic': {'correct': 0, 'total': 0, 'extracted': 0},
        'overall': {'correct': 0, 'total': 0, 'extracted': 0}
    }
    
    detailed_results = []
    
    print("🔬 Evaluating...")
    for prob in tqdm(all_problems):
        # Determine category
        if prob['id'].startswith('add_') or prob['id'].startswith('sub_') or \
           prob['id'].startswith('mul_') or prob['id'].startswith('eq_'):
            category = 'synthetic'
        elif prob['id'].startswith('math_'):
            category = 'math'
        else:
            category = 'reference'
        
        results[category]['total'] += 1
        results['overall']['total'] += 1
        
        # Generate
        try:
            prompt = f"Problem: {prob['problem']}\n\nSolution:"
            encoded = tokenizer.encode(prompt, max_length=512, truncation=True)
            input_ids = torch.tensor([encoded['input_ids']]).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=200,
                    temperature=0.0,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            generated = tokenizer.decode(
                outputs[0][input_ids.shape[1]:].tolist(),
                skip_special_tokens=True
            )
            
            # Extract answer
            predicted = extractor.extract(generated)
            ground_truth = prob['answer']
            
            is_correct = False
            if predicted:
                results[category]['extracted'] += 1
                results['overall']['extracted'] += 1
                is_correct = compare_answers(predicted, ground_truth, tolerance=1e-4)
                if is_correct:
                    results[category]['correct'] += 1
                    results['overall']['correct'] += 1
            
            detailed_results.append({
                'id': prob['id'],
                'category': category,
                'problem': prob['problem'],
                'ground_truth': ground_truth,
                'predicted': predicted,
                'is_correct': is_correct,
                'generated': generated[:300]
            })
            
        except Exception as e:
            print(f"\n   Error on {prob['id']}: {e}")
            detailed_results.append({
                'id': prob['id'],
                'category': category,
                'problem': prob['problem'],
                'ground_truth': prob['answer'],
                'predicted': None,
                'is_correct': False,
                'generated': f'Error: {e}'
            })
    
    # Print results
    print(f"\n{'='*70}")
    print("📊 COMPREHENSIVE RESULTS")
    print(f"{'='*70}\n")
    
    for category in ['reference', 'math', 'synthetic', 'overall']:
        r = results[category]
        if r['total'] > 0:
            accuracy = (r['correct'] / r['total']) * 100
            extraction = (r['extracted'] / r['total']) * 100
            
            print(f"{category.upper()}:")
            print(f"   Total:      {r['total']}")
            print(f"   Correct:    {r['correct']}")
            print(f"   Accuracy:   {accuracy:.1f}%")
            print(f"   Extraction: {extraction:.1f}%")
            print()
    
    # Interpretation
    overall_acc = (results['overall']['correct'] / results['overall']['total']) * 100
    
    print(f"{'='*70}")
    print("📈 INTERPRETATION")
    print(f"{'='*70}\n")
    
    if overall_acc >= 40:
        print("✅ EXCELLENT: Model is performing well!")
        print("   Ready for fine-tuning and competition.")
    elif overall_acc >= 20:
        print("✅ GOOD: Model has learned foundation!")
        print("   Continue training or move to fine-tuning.")
    elif overall_acc >= 10:
        print("⚠️  MODERATE: Model is learning but needs work.")
        print("   Train longer or increase model size.")
    else:
        print("❌ LOW: Model needs more training.")
        print("   Consider: longer training, larger model, or better data.")
    
    # Show examples
    print(f"\n{'='*70}")
    print("📝 EXAMPLE RESULTS")
    print(f"{'='*70}\n")
    
    correct_ex = [r for r in detailed_results if r['is_correct']][:5]
    if correct_ex:
        print("✅ Correct Examples:\n")
        for i, ex in enumerate(correct_ex, 1):
            print(f"{i}. [{ex['category']}] {ex['problem'][:60]}")
            print(f"   Answer: {ex['ground_truth']} | Predicted: {ex['predicted']}\n")
    
    incorrect_ex = [r for r in detailed_results if not r['is_correct']][:5]
    if incorrect_ex:
        print("❌ Incorrect Examples:\n")
        for i, ex in enumerate(incorrect_ex, 1):
            print(f"{i}. [{ex['category']}] {ex['problem'][:60]}")
            print(f"   Answer: {ex['ground_truth']} | Predicted: {ex['predicted']}")
            print(f"   Generated: {ex['generated'][:100]}...\n")
    
    # Save results
    output_dir = Path(checkpoint_path).parent
    results_file = output_dir / "comprehensive_evaluation.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'detailed_results': detailed_results
        }, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")
    print(f"{'='*70}\n")
    
    return overall_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase2/best_model.pt")
    parser.add_argument("--num-problems", type=int, default=100)
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint, args.num_problems)
