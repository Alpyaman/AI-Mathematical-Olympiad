"""Minimal test with debug info"""
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.model.decoder import MathTransformerDecoder
from src.tokenizer.math_tokenizer import MathTokenizer
from src.config.model_config import get_small_config, MathTransformerConfig

torch.serialization.add_safe_globals([MathTransformerConfig])

print("Loading checkpoint...")
checkpoint_path = Path(__file__).parent / "checkpoints" / "phase2" / "best_model.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

vocab_size = checkpoint['model_state_dict']['embed_tokens.weight'].shape[0]
print(f"Vocab size: {vocab_size}")

config = get_small_config()
config.vocab_size = vocab_size

model = MathTransformerDecoder(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded! {sum(p.numel() for p in model.parameters()):,} params")

tokenizer = MathTokenizer()
print(f"Tokenizer vocab: {len(tokenizer)}")
print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"PAD token ID: {tokenizer.pad_token_id}")

# Simple test
problem = "What is 2 + 2?"
prompt = f"Problem: {problem}\n\nSolution:"

encoded = tokenizer.encode(prompt)
input_ids = torch.tensor([encoded['input_ids']], dtype=torch.long)
print(f"\nInput shape: {input_ids.shape}")
print(f"First 10 tokens: {input_ids[0, :10].tolist()}")

print("\nGenerating (max 50 tokens, greedy)...")
try:
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=50,
            temperature=0.0,  # Greedy
            do_sample=False,
            eos_token_id=None,  # Don't check EOS to avoid the error
        )
    
    print(f"Output shape: {outputs.shape}")
    generated_ids = outputs[0, input_ids.shape[1]:].tolist()
    print(f"Generated IDs: {generated_ids[:20]}")
    
    generated_text = tokenizer.decode(generated_ids)
    print(f"\nGenerated text:\n{generated_text}")
    print("\nSUCCESS!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
