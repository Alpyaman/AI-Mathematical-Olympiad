import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset
from tqdm import tqdm
# import wandb
import os

from src.config.model_config import get_small_config
from src.model.decoder import MathTransformerDecoder
from src.tokenizer.math_tokenizer import MathTokenizer
from src.data.dataset import MathReasoningDataset, create_dataloaders, split_dataset
from src.data.data_schema import MathProblem, MathSolution, ReasoningStep, DifficultyLevel, ProblemType

# --- CONFIGURATION ---
BATCH_SIZE = 4        # Decrease if OOM (Out of Memory)
GRAD_ACCUM_STEPS = 4  # Effective Batch Size = 16
LEARNING_RATE = 3e-4
EPOCHS = 10
MAX_LENGTH = 512      # Real math problems need space
# ---------------------

def convert_hf_to_schema(hf_dataset):
    """Converts Hugging Face MATH-500 dataset to our MathProblem schema"""
    problems = []
    print("🔄 Converting dataset...")
    
    for i, item in enumerate(tqdm(hf_dataset)):
        # Construct solution object
        # Note: MATH dataset solutions are just text, so we wrap them in one step for now.
        sol = MathSolution(
            steps=[ReasoningStep(1, "Solution", item['solution'], None)], 
            final_answer=item['answer'],
            answer_type="exact",
            verification=None
        )
        
        prob = MathProblem(
            problem_id=f"MATH_{i}",
            problem_statement=item['problem'],
            solution=sol,
            difficulty=DifficultyLevel.MEDIUM, # Placeholder
            problem_type=ProblemType.ALGEBRA,  # Placeholder
            topics=[item.get('subject', 'math')],
            source="MATH-500",
            year=2024
        )
        problems.append(prob)
    
    return problems

def train():
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Phase 2 Training on {device}")
    
    os.makedirs("checkpoints/phase2", exist_ok=True)
    
    # Initialize WandB
    # wandb.init(project="math-olympiad-ai", name="phase2-math500-run1")

    # 2. Prepare Data
    print("📚 Loading HuggingFaceH4/MATH-500...")
    dataset_hf = load_dataset(" ", split="test") # It's a test set we use for training practice
    problems = convert_hf_to_schema(dataset_hf)
    
    # Split
    train_probs, val_probs, _ = split_dataset(problems, 0.9, 0.1, 0.0)
    print(f"   Train: {len(train_probs)} | Val: {len(val_probs)}")

    # Tokenizer & Dataset
    tokenizer = MathTokenizer()
    train_ds = MathReasoningDataset(train_probs, tokenizer, max_length=MAX_LENGTH)
    val_ds = MathReasoningDataset(val_probs, tokenizer, max_length=MAX_LENGTH)
    
    train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=BATCH_SIZE)

    # 3. Model
    config = get_small_config()
    # config.hidden_dropout = 0.1 # Add dropout for regularization if needed
    model = MathTransformerDecoder(config).to(device)
    
    print(f"   Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 4. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader))

    # 5. Training Loop
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for step, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss'] / GRAD_ACCUM_STEPS # Scale loss
            
            # Backward
            loss.backward()
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log
                current_lr = scheduler.get_last_lr()[0]
                # wandb.log({"train_loss": loss.item() * GRAD_ACCUM_STEPS, "lr": current_lr, "step": global_step})
                pbar.set_postfix(loss=loss.item() * GRAD_ACCUM_STEPS, lr=f"{current_lr:.2e}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs['loss'].item()
        
        avg_val_loss = val_loss / len(val_loader)
        # wandb.log({"val_loss": avg_val_loss, "epoch": epoch+1})
        print(f"   📉 Validation Loss: {avg_val_loss:.4f}")

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/phase2/best_model.pt")
            print("   ✅ New Best Model Saved!")
        
        # Periodic Save
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/phase2/epoch_{epoch+1}.pt")

    print("🏁 Training Complete!")
    # wandb.finish()

if __name__ == "__main__":
    train()