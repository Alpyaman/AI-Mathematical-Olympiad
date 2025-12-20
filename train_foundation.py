"""
Foundation Training Script
Train a tiny model on simple math problems to establish 20%+ baseline accuracy
This is the foundation before fine-tuning.
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
from pathlib import Path

from src.config.model_config import MathTransformerConfig
from src.model.decoder import MathTransformerDecoder
from src.tokenizer.math_tokenizer import MathTokenizer
from src.data.dataset import MathReasoningDataset, create_dataloaders, split_dataset
from src.data.data_loader import MathDatasetLoader
from src.evaluation.answer_extraction import AnswerExtractor

# ============================================================================
# CONFIGURATION - Optimized for getting 20%+ accuracy quickly
# ============================================================================

CONFIG = {
    # Model - Very small for quick training
    "hidden_size": 256,
    "num_layers": 6,
    "num_heads": 8,
    "intermediate_size": 1024,
    "max_length": 512,
    
    # Training
    "batch_size": 4,
    "grad_accum": 4,  # Effective batch = 16
    "learning_rate": 5e-4,
    "epochs": 20,
    "warmup_epochs": 2,
    
    # Data - Start with easy problems
    "max_problems": 1000,  # Limit to prevent overfitting
    "difficulty_filter": ["Level 1", "Level 2"],  # Easy and Medium only
    
    # Checkpointing
    "checkpoint_dir": "checkpoints/foundation",
    "save_every": 5,
}

def create_tiny_model(tokenizer):
    """Create tiny model optimized for learning"""
    config = MathTransformerConfig(
        vocab_size=len(tokenizer),
        hidden_size=CONFIG["hidden_size"],
        num_hidden_layers=CONFIG["num_layers"],
        num_attention_heads=CONFIG["num_heads"],
        intermediate_size=CONFIG["intermediate_size"],
        max_position_embeddings=CONFIG["max_length"],
        max_sequence_length=CONFIG["max_length"],
        hidden_dropout=0.1,
        attention_dropout=0.1,
    )
    
    model = MathTransformerDecoder(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
    
    return model, config

def load_simple_dataset():
    """Load MATH dataset filtered for easier problems"""
    print("\n📚 Loading dataset...")
    loader = MathDatasetLoader()
    
    try:
        # Try loading MATH dataset
        problems = loader.load_math_dataset(
            split="train",
            difficulty_filter=CONFIG["difficulty_filter"]
        )
        print(f"   Loaded {len(problems)} problems from MATH dataset")
    except Exception as e:
        print(f"   Could not load MATH dataset: {e}")
        print("   Creating synthetic dataset...")
        problems = create_synthetic_dataset()
    
    # Limit dataset size
    if len(problems) > CONFIG["max_problems"]:
        problems = problems[:CONFIG["max_problems"]]
        print(f"   Limited to {len(problems)} problems")
    
    return problems

def create_synthetic_dataset():
    """Create simple synthetic math problems if MATH dataset unavailable"""
    from src.data.data_schema import MathProblem, MathSolution, ReasoningStep, DifficultyLevel, ProblemType
    import random
    
    problems = []
    
    # Simple addition problems
    for i in range(200):
        a, b = random.randint(1, 50), random.randint(1, 50)
        answer = a + b
        problems.append(MathProblem(
            problem_id=f"add_{i}",
            problem_statement=f"What is {a} + {b}?",
            solution=MathSolution(
                steps=[ReasoningStep(1, "Addition", f"{a} + {b} = {answer}", None)],
                final_answer=str(answer),
                answer_type="integer"
            ),
            difficulty=DifficultyLevel.EASY,
            problem_type=ProblemType.ALGEBRA,
            topics=["arithmetic"],
            source="synthetic"
        ))
    
    # Simple multiplication
    for i in range(200):
        a, b = random.randint(2, 12), random.randint(2, 12)
        answer = a * b
        problems.append(MathProblem(
            problem_id=f"mul_{i}",
            problem_statement=f"What is {a} × {b}?",
            solution=MathSolution(
                steps=[ReasoningStep(1, "Multiplication", f"{a} × {b} = {answer}", None)],
                final_answer=str(answer),
                answer_type="integer"
            ),
            difficulty=DifficultyLevel.EASY,
            problem_type=ProblemType.ALGEBRA,
            topics=["arithmetic"],
            source="synthetic"
        ))
    
    # Simple linear equations
    for i in range(200):
        a, b, c = random.randint(2, 10), random.randint(1, 20), random.randint(1, 50)
        answer = (c - b) / a
        if answer == int(answer):
            answer = int(answer)
            problems.append(MathProblem(
                problem_id=f"eq_{i}",
                problem_statement=f"Solve for x: {a}x + {b} = {c}",
                solution=MathSolution(
                    steps=[
                        ReasoningStep(1, "Subtract", f"{a}x = {c - b}", None),
                        ReasoningStep(2, "Divide", f"x = {answer}", None),
                    ],
                    final_answer=str(answer),
                    answer_type="integer"
                ),
                difficulty=DifficultyLevel.MEDIUM,
                problem_type=ProblemType.ALGEBRA,
                topics=["linear_equations"],
                source="synthetic"
            ))
    
    print(f"   Generated {len(problems)} synthetic problems")
    return problems

def evaluate_model(model, tokenizer, val_loader, device):
    """Quick evaluation to check if model is learning"""
    model.eval()
    extractor = AnswerExtractor()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if total >= 50:  # Quick eval on 50 problems
                break
            
            input_ids = batch['input_ids'].to(device)
            
            # Generate
            try:
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=100,
                    temperature=0.0,  # Greedy
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Decode and check
                for i in range(outputs.shape[0]):
                    generated = tokenizer.decode(outputs[i].tolist(), skip_special_tokens=True)
                    extracted = extractor.extract(generated)
                    
                    # Get ground truth from batch if available
                    # This is simplified - in real eval you'd need the original problem
                    if extracted:
                        total += 1
                        # For now just count if we extracted something
                        # In real eval, compare with ground truth
            except Exception as e:
                print(f"      Eval error: {e}")
                continue
    
    model.train()
    return total

def train():
    """Main training function"""
    print("\n" + "="*70)
    print("🚀 FOUNDATION MODEL TRAINING")
    print("="*70)
    print("Goal: Achieve 20%+ accuracy on simple math problems")
    print("This will be the foundation for fine-tuning")
    print("="*70 + "\n")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    # Initialize tokenizer
    print("\n🔤 Initializing tokenizer...")
    tokenizer = MathTokenizer()
    print(f"   Vocab size: {len(tokenizer)}")
    
    # Create model
    print("\n  Creating model...")
    model, config = create_tiny_model(tokenizer)
    model = model.to(device)
    
    # Load dataset
    problems = load_simple_dataset()
    
    # Split dataset
    train_probs, val_probs, test_probs = split_dataset(
        problems, 
        train_ratio=0.8,
        val_ratio=0.15,
        test_ratio=0.05
    )
    print("\n📊 Dataset split:")
    print(f"   Train: {len(train_probs)}")
    print(f"   Val:   {len(val_probs)}")
    print(f"   Test:  {len(test_probs)}")
    
    # Create dataloaders
    train_ds = MathReasoningDataset(train_probs, tokenizer, max_length=CONFIG["max_length"])
    val_ds = MathReasoningDataset(val_probs, tokenizer, max_length=CONFIG["max_length"])
    
    train_loader, val_loader = create_dataloaders(
        train_ds, val_ds,
        batch_size=CONFIG["batch_size"],
        num_workers=0  # 0 for Windows compatibility
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=0.01
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["epochs"],
        eta_min=1e-6
    )
    
    # Training loop
    print(f"\n{'='*70}")
    print("🎓 TRAINING STARTED")
    print(f"{'='*70}\n")
    
    best_val_loss = float('inf')
    checkpoint_dir = Path(CONFIG["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        steps = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        optimizer.zero_grad()
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss'] / CONFIG["grad_accum"]
            
            # Backward
            loss.backward()
            
            if (step + 1) % CONFIG["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                steps += 1
            
            total_loss += loss.item() * CONFIG["grad_accum"]
            pbar.set_postfix({'loss': f"{loss.item() * CONFIG['grad_accum']:.4f}"})
        
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
        
        scheduler.step()
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} Complete")
        print(f"{'='*70}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")
        print(f"LR:         {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config,
            }, checkpoint_dir / "best_model.pt")
            print(f"✅ Saved best model (val_loss: {avg_val_loss:.4f})")
        
        # Periodic checkpoint
        if (epoch + 1) % CONFIG["save_every"] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config,
            }, checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")
            print(f"💾 Saved checkpoint (epoch {epoch+1})")
        
        print(f"{'='*70}\n")
    
    print(f"\n{'='*70}")
    print("🏁 TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("\n📝 Next step: Evaluate on test set")
    print(f"{'='*70}\n")
    
    # Save training config
    with open(checkpoint_dir / "training_config.json", 'w') as f:
        json.dump(CONFIG, f, indent=2)

if __name__ == "__main__":
    train()
