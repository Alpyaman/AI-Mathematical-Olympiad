"""
Phase 1.1 Demo: Mathematical Reasoning Transformer

This script demonstrates the customized decoder-only transformer architecture
with features optimized for mathematical reasoning:
1. Extended context length for long proofs
2. RoPE positional encoding with length generalization
3. Specialized tokenizer for mathematical notation
"""

import torch
from src import (
    MathTransformerConfig,
    MathTransformerDecoder,
    MathTokenizer,
    get_small_config,
    get_base_config,
)


def demo_tokenizer():
    """Demonstrate the mathematical tokenizer capabilities."""
    print("\n" + "="*70)
    print("PHASE 1.1 - MATHEMATICAL TOKENIZER DEMONSTRATION")
    print("="*70)

    tokenizer = MathTokenizer()

    # Mathematical problems in various notations
    test_problems = [
        "Prove that for all natural numbers n, the sum 1 + 2 + ... + n = n(n+1)/2",
        "∀x ∈ ℝ, if x² = 4 then x = 2 or x = -2",
        "Let f: ℝ → ℝ be defined by f(x) = ∫₀ˣ e^(-t²) dt. Prove f is strictly increasing.",
        "Prove: ∑_{k=1}^{∞} 1/k² = π²/6",
        "If α and β are roots of x² + px + q = 0, then α + β = -p and αβ = q",
    ]

    print("\nTokenizing mathematical problems:\n")
    for i, problem in enumerate(test_problems, 1):
        encoded = tokenizer.encode(problem, padding=False)
        decoded = tokenizer.decode(encoded["input_ids"])

        print(f"{i}. Original ({len(problem)} chars, {len(encoded['input_ids'])} tokens):")
        print(f"   {problem}")
        print(f"   Decoded: {decoded}")
        print()

    return tokenizer


def demo_model_architecture(tokenizer):
    """Demonstrate the model architecture and its capabilities."""
    print("\n" + "="*70)
    print("PHASE 1.1 - MODEL ARCHITECTURE DEMONSTRATION")
    print("="*70)

    # Create a small configuration for demonstration
    config = get_small_config()

    print("\nModel Configuration:")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Num Layers: {config.num_hidden_layers}")
    print(f"  Num Attention Heads: {config.num_attention_heads}")
    print(f"  Max Context Length: {config.max_position_embeddings}")
    print(f"  Vocab Size: {config.vocab_size}")
    print(f"  Intermediate Size: {config.intermediate_size}")
    print(f"  RoPE Base: {config.rope_theta}")
    print(f"  Dynamic RoPE: {config.use_dynamic_rope}")

    # Initialize model
    print("\nInitializing model...")
    model = MathTransformerDecoder(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Model Size: ~{total_params * 4 / (1024**2):.2f} MB (fp32)")

    return model, config


def demo_forward_pass(model, tokenizer, config):
    """Demonstrate a forward pass through the model."""
    print("\n" + "="*70)
    print("PHASE 1.1 - FORWARD PASS DEMONSTRATION")
    print("="*70)

    # Create a sample mathematical problem
    problem = "Prove that ∀n ∈ ℕ, n² + n is even."

    print(f"\nInput Problem: {problem}\n")

    # Encode
    encoded = tokenizer.encode(problem, padding=True, max_length=128)
    input_ids = torch.tensor([encoded["input_ids"]])
    attention_mask = torch.tensor([encoded["attention_mask"]])

    print(f"Encoded shape: {input_ids.shape}")
    print(f"Sequence length: {input_ids.shape[1]}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    logits = outputs["logits"]
    print(f"\nOutput logits shape: {logits.shape}")
    print(f"  Batch size: {logits.shape[0]}")
    print(f"  Sequence length: {logits.shape[1]}")
    print(f"  Vocab size: {logits.shape[2]}")

    # Get predicted tokens
    predicted_ids = torch.argmax(logits, dim=-1)
    print(f"\nPredicted token IDs shape: {predicted_ids.shape}")


def demo_generation(model, tokenizer):
    """Demonstrate text generation."""
    print("\n" + "="*70)
    print("PHASE 1.1 - GENERATION DEMONSTRATION")
    print("="*70)

    # Note: This is with an untrained model, so output will be random
    problem = "Prove: 1 + 1 ="

    print(f"\nPrompt: {problem}")
    print("Note: Model is untrained, so generation will be random.\n")

    # Encode prompt
    encoded = tokenizer.encode(problem, add_special_tokens=True, padding=False)
    input_ids = torch.tensor([encoded["input_ids"]])

    # Generate (with untrained model, just for demonstration)
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=20,
            temperature=1.0,
            do_sample=True,
        )

    # Decode
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(f"Generated: {generated_text}")
    print("\nTo get meaningful results, the model needs to be trained on mathematical data.")


def demo_rope_scaling():
    """Demonstrate RoPE length generalization."""
    print("\n" + "="*70)
    print("PHASE 1.1 - RoPE LENGTH GENERALIZATION DEMONSTRATION")
    print("="*70)

    from src.model.rope import RotaryPositionalEmbedding, DynamicRoPE

    # Standard RoPE
    rope = RotaryPositionalEmbedding(
        dim=64,
        max_position_embeddings=2048,
        base=10000.0,
    )

    # Dynamic RoPE
    dynamic_rope = DynamicRoPE(
        dim=64,
        max_position_embeddings=2048,
        base=10000.0,
        max_scaling_factor=4.0,
    )

    print("\nStandard RoPE Configuration:")
    print(f"  Dimension: {rope.dim}")
    print(f"  Max Position Embeddings: {rope.max_position_embeddings}")
    print(f"  Base: {rope.base}")

    print("\nDynamic RoPE Configuration:")
    print(f"  Dimension: {dynamic_rope.dim}")
    print(f"  Max Position Embeddings: {dynamic_rope.max_position_embeddings}")
    print(f"  Max Scaling Factor: {dynamic_rope.max_scaling_factor}")

    # Test with sequences of different lengths
    test_lengths = [512, 2048, 4096, 8192]

    print("\nTesting RoPE with different sequence lengths:")
    for seq_len in test_lengths:
        # Create dummy query and key tensors
        q = torch.randn(1, 8, seq_len, 64)  # (batch, heads, seq_len, dim)
        k = torch.randn(1, 8, seq_len, 64)

        # Apply dynamic RoPE
        q_rot, k_rot = dynamic_rope(q, k)

        extrapolation = "✓ (extrapolating)" if seq_len > 2048 else "  (within range)"
        scaling = dynamic_rope.scaling_factor if seq_len > 2048 else 1.0

        print(f"  Seq length {seq_len:5d}: scaling_factor={scaling:.2f} {extrapolation}")


def print_phase_summary():
    """Print Phase 1.1 summary."""
    print("\n" + "="*70)
    print("PHASE 1.1 IMPLEMENTATION SUMMARY")
    print("="*70)

    summary = """
Phase 1.1: Initial Architecture Design - COMPLETED ✓

Key Components Implemented:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Decoder-Only Transformer Architecture
   • Based on proven Llama/GPT architecture
   • Multi-head attention with Grouped-Query Attention (GQA)
   • SwiGLU feed-forward networks
   • RMSNorm for efficient normalization
   • Gradient checkpointing support

2. Positional Encoding
   • Rotary Position Embeddings (RoPE)
   • Dynamic RoPE scaling for length generalization
   • Supports extrapolation to 2-4x training length
   • Efficient caching for inference

3. Extended Context Length
   • Default: 8192 tokens (small config: 2048, large config: 16384)
   • Sufficient for multi-step mathematical proofs
   • KV caching for efficient long-context generation

4. Mathematical Tokenizer
   • Handles 100+ mathematical symbols (∀, ∃, ∑, ∫, etc.)
   • LaTeX notation support
   • Special tokens for mathematical structures
   • Character-level fallback for unknown symbols

5. Configuration System
   • Three preset configs: small (~125M), base (~1B), large (~7B)
   • Flexible hyperparameter tuning
   • Easy to extend and customize

Architecture Highlights:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Optimized for mathematical reasoning tasks
• Length generalization via dynamic RoPE
• Efficient inference with KV caching
• Ready for Phase 1.2: Dataset preparation and training

Next Steps (Phase 1.2):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Collect and curate mathematical olympiad datasets
• Implement data preprocessing pipeline
• Set up training infrastructure
• Define evaluation metrics
"""
    print(summary)


def main():
    """Run all Phase 1.1 demonstrations."""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "AI MATHEMATICAL OLYMPIAD" + " "*29 + "║")
    print("║" + " "*10 + "Phase 1.1: Initial Architecture Design" + " "*20 + "║")
    print("╚" + "═"*68 + "╝")

    # 1. Tokenizer demo
    tokenizer = demo_tokenizer()

    # 2. Model architecture demo
    model, config = demo_model_architecture(tokenizer)

    # 3. Forward pass demo
    demo_forward_pass(model, tokenizer, config)

    # 4. Generation demo
    demo_generation(model, tokenizer)

    # 5. RoPE scaling demo
    demo_rope_scaling()

    # 6. Summary
    print_phase_summary()

    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()