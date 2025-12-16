"""
Quick test script for Phase 1.1 implementation.

Run this to verify that the architecture is working correctly.
"""

import torch
from src import (
    MathTransformerConfig,
    MathTransformerDecoder,
    MathTokenizer,
    get_small_config,
)


def test_tokenizer():
    """Test mathematical tokenizer."""
    print("Testing Mathematical Tokenizer...")

    tokenizer = MathTokenizer()

    # Test basic encoding/decoding
    text = "∀x ∈ ℝ, x² ≥ 0"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded["input_ids"])

    assert isinstance(encoded["input_ids"], list), "Encoded should be a list"
    assert isinstance(decoded, str), "Decoded should be a string"
    assert len(encoded["input_ids"]) > 0, "Should have tokens"

    print(f"  ✓ Input: {text}")
    print(f"  ✓ Tokens: {len(encoded['input_ids'])}")
    print(f"  ✓ Decoded: {decoded}")
    print("  ✓ Tokenizer working correctly\n")

    return tokenizer


def test_model():
    """Test model initialization and forward pass."""
    print("Testing Model Architecture...")

    config = get_small_config()
    model = MathTransformerDecoder(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model initialized with {total_params:,} parameters")

    # Test forward pass
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)
    print("  ✓ Forward pass successful")
    print(f"  ✓ Output shape: {outputs['logits'].shape}\n")

    return model


def test_generation(model, tokenizer):
    """Test text generation."""
    print("Testing Text Generation...")

    text = "Prove: 1 + 1 ="
    encoded = tokenizer.encode(text, add_special_tokens=True, padding=False)
    input_ids = torch.tensor([encoded["input_ids"]])

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=10,
            temperature=1.0,
            do_sample=False,  # Greedy for deterministic test
        )

    assert generated_ids.shape[1] > input_ids.shape[1], "Should generate new tokens"
    print(f"  ✓ Generated {generated_ids.shape[1] - input_ids.shape[1]} new tokens")
    print("  ✓ Generation working correctly\n")


def test_rope():
    """Test RoPE implementation."""
    print("Testing RoPE Positional Encoding...")

    from src.model.rope import RotaryPositionalEmbedding, DynamicRoPE

    # Test standard RoPE
    rope = RotaryPositionalEmbedding(dim=64, max_position_embeddings=2048)

    batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    q_rot, k_rot = rope(q, k)

    assert q_rot.shape == q.shape, "Query shape should be preserved"
    assert k_rot.shape == k.shape, "Key shape should be preserved"
    print("  ✓ Standard RoPE working correctly")

    # Test dynamic RoPE
    dynamic_rope = DynamicRoPE(dim=64, max_position_embeddings=2048)
    q_rot, k_rot = dynamic_rope(q, k)

    assert q_rot.shape == q.shape, "Query shape should be preserved"
    assert k_rot.shape == k.shape, "Key shape should be preserved"
    print("  ✓ Dynamic RoPE working correctly")
    print("  ✓ RoPE tests passed\n")


def test_kv_cache(model):
    """Test KV caching."""
    print("Testing KV Cache...")

    batch_size, seq_len = 1, 32
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

    model.eval()
    with torch.no_grad():
        # First pass with cache
        outputs1 = model(input_ids=input_ids, use_cache=True)
        past_kv = outputs1["past_key_values"]

        assert past_kv is not None, "Should return cache"
        assert len(past_kv) == model.config.num_hidden_layers, "Should have cache for each layer"

        # Second pass with cache
        new_token = torch.randint(0, model.config.vocab_size, (batch_size, 1))
        outputs2 = model(input_ids=new_token, past_key_values=past_kv, use_cache=True)

        assert outputs2["logits"].shape[1] == 1, "Should only output for new token"
        print("  ✓ KV cache working correctly")
        print(f"  ✓ Cache has {len(past_kv)} layers\n")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Phase 1.1: Architecture Tests")
    print("="*60 + "\n")

    try:
        # Test components
        tokenizer = test_tokenizer()
        model = test_model()
        test_generation(model, tokenizer)
        test_rope()
        test_kv_cache(model)

        print("="*60)
        print("✓ All tests passed successfully!")
        print("="*60 + "\n")
        print("Phase 1.1 implementation is working correctly.")
        print("Ready to proceed to Phase 1.2: Dataset Preparation\n")

        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)