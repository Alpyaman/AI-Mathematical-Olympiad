import torch
from torch.optim import AdamW
from src.config.model_config import get_small_config
from src.model.decoder import MathTransformerDecoder
from src.tokenizer.math_tokenizer import MathTokenizer
from src.data.dataset import MathReasoningDataset, create_dataloaders
from src.data.data_schema import MathProblem, MathSolution, ReasoningStep, DifficultyLevel, ProblemType
from src.data.data_formatter import ChainOfThoughtFormatter # Import Formatter

def run_overfit_check():
    print("🔬 STARTING FINAL OVERFIT CHECK...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Create Problem
    sol = MathSolution(
        steps=[
            ReasoningStep(1, "Subtract 10", "3x = 15", None),
            ReasoningStep(2, "Divide by 3", "x = 5", None)
        ], 
        final_answer="x = 5", answer_type="exact", verification=None
    )
    problem = MathProblem(
        "ID", "Solve for x: 3x + 10 = 25", sol, 
        DifficultyLevel.EASY, ProblemType.ALGEBRA, [], "src", 2024
    )
    
    # 2. Setup
    tokenizer = MathTokenizer()
    # Ensure tokenizer has the fix
    if tokenizer.encode("\n", add_special_tokens=False)["input_ids"] == []:
        print("⚠️ WARNING: Tokenizer still stripping newlines! Please apply Fix 1.")
        
    config = get_small_config()
    dataset = MathReasoningDataset([problem], tokenizer, max_length=128)
    loader = create_dataloaders(dataset, batch_size=1)
    model = MathTransformerDecoder(config).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # 3. Train
    model.train()
    for i in range(150):
        batch = next(iter(loader))
        loss = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels=batch['labels'].to(device))['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 25 == 0:
            print(f"   Step {i}: Loss {loss.item():.6f}")

    # 4. Inference using FORMATTER to guarantee match
    model.eval()
    formatter = ChainOfThoughtFormatter(use_special_tokens=True)
    # Generate prompt exactly like training data, but stop before solution steps
    full_text = formatter.format_problem(problem)
    # Split at <solution> to simulate inference state
    prompt_text = full_text.split("<solution>")[0] + "<solution>"
    
    print(f"\n📝 Prompt: {repr(prompt_text)}")
    
    input_ids = torch.tensor(tokenizer.encode(prompt_text, add_special_tokens=False)['input_ids']).unsqueeze(0).to(device)
    
    output_ids = model.generate(
        input_ids, 
        max_new_tokens=100, 
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )
    
    decoded = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=False)
    print("\n✨ RESULT:")
    print(decoded)

if __name__ == "__main__":
    run_overfit_check()