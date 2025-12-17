import torch
from src.config.model_config import get_small_config
from src.model.decoder import MathTransformerDecoder
from src.tokenizer.math_tokenizer import MathTokenizer

def generate_solution(problem_text):
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")

    # 2. Load Model & Tokenizer
    # Important: Must use same config as training!
    config = get_small_config() 
    model = MathTransformerDecoder(config).to(device)
    
    # Load the best weights you just trained
    checkpoint_path = "checkpoints/math500_best.pt"
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("✅ Weights loaded successfully!")
    except FileNotFoundError:
        print("❌ Checkpoint not found. Make sure you ran train_math_real.py first.")
        return

    tokenizer = MathTokenizer()

    # 3. Prepare Input
    # We format it exactly how the model expects (Standard CoT format)
    input_text = f"<bos>\nProblem: {problem_text}\n\n<solution>\n"
    print(f"\n📝 Input Prompt:\n{input_text.strip()}")
    
    input_ids = torch.tensor(tokenizer.encode(input_text)['input_ids']).unsqueeze(0).to(device)

    # 4. Generate
    print("\n🧠 Thinking...")
    output_ids = model.generate(
        input_ids, 
        max_new_tokens=200, 
        temperature=1.0, # Creativity factor
        do_sample=False
    )

    # 5. Decode
    generated_text = tokenizer.decode(output_ids[0].tolist())
    print("\n✨ Generated Solution:")
    print("--------------------------------------------------")
    print(generated_text)
    print("--------------------------------------------------")

if __name__ == "__main__":
    # A simple algebra problem similar to what it saw in training
    test_problem = "Solve for x: 3x + 10 = 25"
    generate_solution(test_problem)