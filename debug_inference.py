import torch
from src.config.model_config import get_small_config
from src.model.decoder import MathTransformerDecoder
from src.tokenizer.math_tokenizer import MathTokenizer

def diagnose():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🕵️ Diagnosing on {device}...")

    # 1. Load Model
    tokenizer = MathTokenizer()
    config = get_small_config()
    model = MathTransformerDecoder(config).to(device)
    
    # Load your overfit model (Low Loss Model)
    # Note: If train_overfit.py didn't save a file, we might need to rely on what's in memory or re-run training.
    # Assuming train_overfit.py trained but didn't save, we will QUICKLY re-train 10 steps here to get state.
    # OR if you have a checkpoint, load it.
    
    # Let's just do a quick re-train of 20 steps to get the state back, 
    # since train_overfit.py doesn't save to disk by default in the script I gave you.
    print("⚡ Quick Re-Training to restore state...")
    from src.data.dataset import MathReasoningDataset, create_dataloaders
    from src.data.data_schema import MathProblem, MathSolution, ReasoningStep, DifficultyLevel, ProblemType
    from torch.optim import AdamW

    sol = MathSolution(steps=[ReasoningStep(1, "Subtract 10", "3x = 15", None)], final_answer="x = 5", answer_type="exact", verification=None)
    prob = MathProblem("ID", "Solve for x: 3x + 10 = 25", sol, DifficultyLevel.EASY, ProblemType.ALGEBRA, [], "src", 2024)
    dataset = MathReasoningDataset([prob], tokenizer, max_length=128)
    loader = create_dataloaders(dataset, batch_size=1)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    model.train()
    for i in range(30):
        batch = next(iter(loader))
        loss = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels=batch['labels'].to(device))['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"   Restored Loss: {loss.item():.4f}")

    # 2. Inference Diagnosis
    model.eval()
    print("\n🔍 INFERENCE DEBUG:")
    input_text = "<bos>\nProblem: Solve for x: 3x + 10 = 25\n\n<solution>\n"
    input_ids = torch.tensor(tokenizer.encode(input_text)['input_ids']).unsqueeze(0).to(device)
    
    # Use the FIXED generate with explicit EOS
    output_ids = model.generate(
        input_ids, 
        max_new_tokens=50, 
        do_sample=False, 
        eos_token_id=tokenizer.eos_token_id # Pass the correct EOS ID
    )
    
    # 3. Analysis
    generated_ids = output_ids[0].tolist()
    new_tokens = generated_ids[len(input_ids[0]):]
    
    print(f"\nRAW INPUT IDs: {input_ids[0].tolist()}")
    print(f"RAW GEN IDs:   {new_tokens}")
    
    print("\n📜 DECODED (skip_special_tokens=False):")
    print(tokenizer.decode(generated_ids, skip_special_tokens=False))
    
    print("\n📜 DECODED (skip_special_tokens=True):")
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))

if __name__ == "__main__":
    diagnose()