"""
Quick Test Script - Test your trained checkpoints

Usage:
    python quick_test.py
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.model.decoder import MathTransformerDecoder
from src.tokenizer.math_tokenizer import MathTokenizer
from src.config.model_config import MathTransformerConfig


def quick_test_checkpoint(checkpoint_path: str, config_name: str = "small"):
    """Quick test of a checkpoint."""
    print(f"\n{'='*80}")
    print(f"🧪 Testing Checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    try:
        # Load config
        config = MathTransformerConfig.get_config(config_name)
        
        # Initialize tokenizer
        tokenizer = MathTokenizer()
        config.vocab_size = len(tokenizer)
        
        # Initialize model
        model = MathTransformerDecoder(config)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded model_state_dict from checkpoint")
            if 'epoch' in checkpoint:
                print(f"   Epoch: {checkpoint['epoch']}")
            if 'train_loss' in checkpoint:
                print(f"   Training Loss: {checkpoint['train_loss']:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Loaded state_dict from checkpoint")
        
        model.eval()
        
        # Test parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("\n📊 Model Info:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Config: {config_name}")
        print(f"   Vocab size: {config.vocab_size}")
        
        # Test generation on simple problem
        test_problem = "What is 2 + 2?"
        print(f"\n🧮 Testing generation on: '{test_problem}'")
        
        prompt = f"Problem: {test_problem}\n\nSolution:"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        print("\n📝 Generated Solution:")
        print(f"   {generated_text[:200]}...")
        
        # Try to extract answer
        import re
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', generated_text)
        if boxed_match:
            answer = boxed_match.group(1)
            print(f"\n✅ Extracted Answer: {answer}")
        else:
            print("\n No boxed answer found in generation")
        
        print(f"\n{'='*80}")
        print("✅ Checkpoint test completed successfully!")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all available checkpoints."""
    print("\n🔍 Discovering available checkpoints...\n")
    
    checkpoints = [
        ("checkpoints/baseline/best_model.pt", "small"),
        ("checkpoints/baseline/checkpoint_epoch_10.pt", "small"),
        ("checkpoints/pretraining_notebook/final.pt", "small"),
        ("checkpoints/pretraining_notebook/final_notebook.pt", "small"),
    ]
    
    print("Found checkpoints:")
    for i, (path, config) in enumerate(checkpoints, 1):
        if Path(path).exists():
            print(f"   {i}. {path} (config: {config})")
    
    print("\nTesting each checkpoint...\n")
    
    results = {}
    for checkpoint_path, config_name in checkpoints:
        if Path(checkpoint_path).exists():
            success = quick_test_checkpoint(checkpoint_path, config_name)
            results[checkpoint_path] = "✅ PASS" if success else "❌ FAIL"
        else:
            print(f"\n⚠️ Checkpoint not found: {checkpoint_path}\n")
            results[checkpoint_path] = "❌ NOT FOUND"
    
    # Summary
    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)
    for checkpoint, result in results.items():
        print(f"   {result} - {checkpoint}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
