"""
Curriculum Learning - Start Simple, Build Up
Trains model on progressively harder math problems
Stage 1: Basic arithmetic (addition, subtraction, multiplication)
Stage 2: Multi-step arithmetic
Stage 3: Simple algebra
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
import random
from pathlib import Path

from src.config.model_config import MathTransformerConfig
from src.model.decoder import MathTransformerDecoder
from src.tokenizer.math_tokenizer import MathTokenizer
from src.data.dataset import MathReasoningDataset, create_dataloaders, split_dataset
from src.data.data_schema import MathProblem, MathSolution, ReasoningStep, DifficultyLevel, ProblemType
from src.data.data_formatter import ChainOfThoughtFormatter

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Model - Medium model (~350M params) for better capacity
    "hidden_size": 768,        # GPT-2 medium size
    "num_layers": 12,          # Deeper network
    "num_heads": 12,           # More attention heads
    "intermediate_size": 3072, # 4x hidden size
    "max_length": 512,         # Longer sequences for complex reasoning
    
    # Training
    "batch_size": 2,           # Small batch for large model
    "grad_accum": 16,          # High accumulation to maintain effective batch of 32
    "learning_rate": 2e-4,     # Lower LR for large model stability
    "epochs_per_stage": 20,    # More epochs for thorough learning
    
    # Curriculum stages
    "problems_per_stage": 2000,
    
    # Output
    "checkpoint_dir": "checkpoints/curriculum",
}

def create_stage1_arithmetic(num_problems=2000, tokenizer=None):
    """Stage 1: Basic single-operation arithmetic"""
    problems = []
    random.seed(42)

    operations = [
        ('addition', lambda a, b: a + b, '+'),
        ('subtraction', lambda a, b: a - b, '-'),
        ('multiplication', lambda a, b: a * b, '×'),
    ]

    for i in range(num_problems):
        op_name, op_func, op_symbol = random.choice(operations)

        if op_name == 'addition':
            a, b = random.randint(1, 50), random.randint(1, 50)
        elif op_name == 'subtraction':
            a, b = random.randint(20, 100), random.randint(1, 19)
        else:  # multiplication
            a, b = random.randint(2, 12), random.randint(2, 12)

        answer = op_func(a, b)

        problems.append(MathProblem(
            problem_id=f"s1_{i}",
            problem_statement=f"What is {a} {op_symbol} {b}?",
            solution=MathSolution(
                steps=[ReasoningStep(1, op_name.title(),
                      f"{a} {op_symbol} {b} = {answer}", None)],
                final_answer=str(answer),
                answer_type="integer"
            ),
            difficulty=DifficultyLevel.EASY,
            problem_type=ProblemType.ALGEBRA,
            topics=["arithmetic"],
            source="curriculum_stage1"
        ))

    return problems

def create_stage2_multistep(num_problems=2000, tokenizer=None):
    """Stage 2: Two-step arithmetic"""
    problems = []
    random.seed(43)

    for i in range(num_problems):
        # a + b - c or a * b + c
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

def create_stage3_algebra(num_problems=2000, tokenizer=None):
    """Stage 3: Simple linear equations"""
    problems = []
    random.seed(44)

    for i in range(num_problems):
        # ax + b = c, solve for x
        a = random.randint(2, 10)
        b = random.randint(1, 20)
        x = random.randint(1, 20)
        c = a * x + b

        problems.append(MathProblem(
            problem_id=f"s3_{i}",
            problem_statement=f"Solve for x: {a}x + {b} = {c}",
            solution=MathSolution(
                steps=[
                    ReasoningStep(1, "Subtract", f"{a}x = {c} - {b} = {c-b}", None),
                    ReasoningStep(2, "Divide", f"x = {c-b} / {a} = {x}", None),
                ],
                final_answer=str(x),
                answer_type="integer"
            ),
            difficulty=DifficultyLevel.MEDIUM,
            problem_type=ProblemType.ALGEBRA,
            topics=["linear_equations"],
            source="curriculum_stage3"
        ))

    return problems

def evaluate_quick(model, tokenizer, problems, device, num_samples=50):
    """Quick evaluation on sample problems"""
    from src.evaluation.answer_extraction import AnswerExtractor, compare_answers

    model.eval()
    extractor = AnswerExtractor()

    correct = 0
    extracted = 0
    sample_problems = random.sample(problems, min(num_samples, len(problems)))

    with torch.no_grad():
        for prob in sample_problems:
            try:
                # Use simple format matching training
                prompt = f"Problem: {prob.problem_statement}\n\nSolution:"
                encoded = tokenizer.encode(prompt, add_special_tokens=False, max_length=256, truncation=True)
                input_ids = torch.tensor([encoded['input_ids']]).to(device)

                outputs = model.generate(
                    input_ids,
                    max_new_tokens=100,
                    temperature=0.0,
                    eos_token_id=tokenizer.eos_token_id,
                )

                generated = tokenizer.decode(
                    outputs[0][input_ids.shape[1]:].tolist(),
                    skip_special_tokens=True
                )

                predicted = extractor.extract(generated)
                ground_truth = prob.solution.final_answer

                if predicted:
                    extracted += 1
                    if compare_answers(predicted, ground_truth, tolerance=1e-4):
                        correct += 1
            except Exception:
                continue

    model.train()

    accuracy = (correct / num_samples) * 100 if num_samples > 0 else 0
    extraction = (extracted / num_samples) * 100 if num_samples > 0 else 0

    return accuracy, extraction

def train_stage(stage_name, problems, model, tokenizer, optimizer, scheduler, device, config):
    """Train on one curriculum stage"""
    print(f"\n{'='*70}")
    print(f"📚 STAGE: {stage_name}")
    print(f"{'='*70}")
    print(f"Problems: {len(problems)}")

    # Split data
    train_probs, val_probs, _ = split_dataset(problems, 0.8, 0.15, 0.05)
    print(f"Train: {len(train_probs)}, Val: {len(val_probs)}")

    # CRITICAL FIX: Create formatter WITHOUT special tokens - just simple Q&A format
    # add_eos_token=True ensures EOS token is added at the END of the formatted text
    simple_formatter = ChainOfThoughtFormatter(
        use_special_tokens=False,  # No <step>, <answer> etc - this was causing 0% accuracy!
        include_step_numbers=False,
        include_justifications=False,
        add_verification=False,
        add_eos_token=True  # Add EOS at the end of the formatted text
    )

    # Create datasets with simple formatter
    train_ds = MathReasoningDataset(
        train_probs, tokenizer,
        formatter=simple_formatter,  # Use simple formatter
        max_length=config["max_length"]
    )
    val_ds = MathReasoningDataset(
        val_probs, tokenizer,
        formatter=simple_formatter,  # Use simple formatter
        max_length=config["max_length"]
    )

    train_loader, val_loader = create_dataloaders(
        train_ds, val_ds,
        batch_size=config["batch_size"],
        num_workers=0
    )

    best_val_loss = float('inf')

    for epoch in range(config["epochs_per_stage"]):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs_per_stage']}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss'] / config["grad_accum"]
            loss.backward()

            if (step + 1) % config["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * config["grad_accum"]
            pbar.set_postfix({'loss': f"{loss.item() * config['grad_accum']:.4f}"})

        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs['loss'].item()
                val_steps += 1

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / val_steps

        # Quick accuracy check
        accuracy, extraction = evaluate_quick(model, tokenizer, val_probs, device, num_samples=30)

        print(f"\nEpoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.1f}%, "
              f"Extraction: {extraction:.1f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"   ✅ Best model for {stage_name}")

    return best_val_loss

def main():
    print("\n" + "="*70)
    print("🎓 CURRICULUM LEARNING - SIMPLE TO COMPLEX")
    print("="*70 + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Initialize tokenizer
    tokenizer = MathTokenizer()

    # Create model
    config = MathTransformerConfig(
        vocab_size=len(tokenizer),
        hidden_size=CONFIG["hidden_size"],
        num_hidden_layers=CONFIG["num_layers"],
        num_attention_heads=CONFIG["num_heads"],
        intermediate_size=CONFIG["intermediate_size"],
        max_position_embeddings=CONFIG["max_length"],
        max_sequence_length=CONFIG["max_length"],
    )

    model = MathTransformerDecoder(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters\n")

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=0.01)
    total_epochs = CONFIG["epochs_per_stage"] * 3  # 3 stages
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs * 100, eta_min=1e-6)

    # Create checkpoint directory
    checkpoint_dir = Path(CONFIG["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Generate curriculum data
    print("📚 Generating curriculum data...\n")
    stage1_problems = create_stage1_arithmetic(CONFIG["problems_per_stage"], tokenizer=tokenizer)
    stage2_problems = create_stage2_multistep(CONFIG["problems_per_stage"], tokenizer=tokenizer)
    stage3_problems = create_stage3_algebra(CONFIG["problems_per_stage"], tokenizer=tokenizer)

    print(f"Stage 1 (Arithmetic): {len(stage1_problems)} problems")
    print(f"Stage 2 (Multi-step): {len(stage2_problems)} problems")
    print(f"Stage 3 (Algebra):    {len(stage3_problems)} problems")

    # Show example of how data will be formatted
    print("\n📝 Example training format (EOS token added at end):")
    print(f"{'='*70}")
    simple_formatter = ChainOfThoughtFormatter(
        use_special_tokens=False,
        include_step_numbers=False,
        include_justifications=False,
        add_verification=False,
        add_eos_token=True,  # EOS at the end
    )
    sample_text = simple_formatter.format_problem(stage1_problems[0], include_solution=True)
    print(sample_text)
    print(f"{'='*70}")
    print("☝️  EOS token is added at the END after 'Final Answer: X'")
    print(f"{'='*70}\n")

    # Train each stage
    stages = [
        ("Stage 1: Basic Arithmetic", stage1_problems),
        ("Stage 2: Multi-step Arithmetic", stage2_problems),
        ("Stage 3: Simple Algebra", stage3_problems),
    ]

    results = {}

    for stage_name, problems in stages:
        val_loss = train_stage(stage_name, problems, model, tokenizer,
                              optimizer, scheduler, device, CONFIG)

        # Save checkpoint after each stage
        stage_num = stage_name.split()[1].rstrip(':')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': config,
            'stage': stage_name,
        }, checkpoint_dir / f"stage{stage_num}_complete.pt")

        results[stage_name] = val_loss

        print(f"\n✅ {stage_name} complete! Val loss: {val_loss:.4f}")

    # Final evaluation
    print(f"\n{'='*70}")
    print("🎯 FINAL EVALUATION")
    print(f"{'='*70}\n")

    # Test on all stages
    all_test = stage1_problems[-200:] + stage2_problems[-200:] + stage3_problems[-200:]
    final_acc, final_ext = evaluate_quick(model, tokenizer, all_test, device, num_samples=100)

    print(f"Final Test Accuracy:  {final_acc:.1f}%")
    print(f"Answer Extraction:    {final_ext:.1f}%")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'val_loss': val_loss,
        'final_accuracy': final_acc,
    }, checkpoint_dir / "curriculum_final.pt")

    # Save metadata
    metadata = {
        'model': {
            'parameters': total_params,
            'config': {
                'hidden_size': CONFIG["hidden_size"],
                'num_layers': CONFIG["num_layers"],
            }
        },
        'training': {
            'problems_per_stage': CONFIG["problems_per_stage"],
            'epochs_per_stage': CONFIG["epochs_per_stage"],
            'stages': list(results.keys()),
        },
        'results': {
            'final_accuracy': final_acc,
            'final_extraction': final_ext,
            'stage_losses': results,
        }
    }

    with open(checkpoint_dir / "curriculum_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print("✅ CURRICULUM TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Final model saved to: {checkpoint_dir / 'curriculum_final.pt'}")
    print(f"Expected accuracy on simple problems: {final_acc:.1f}%")

    if final_acc >= 20:
        print("\n🎉 SUCCESS! Model learned basic math reasoning!")
        print("   Ready to try more complex problems or fine-tuning.")
    else:
        print("\n⚠️  Model needs more training or capacity.")

    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
