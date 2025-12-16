"""
Model Configuration for Mathematical Reasoning Transformer
Phase 1.1: Initial architecture Design
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MathTransformerConfig:
    """
    Configuration class for the mathematical reasoning transformer.

    This configuration is optimized for:
    - Extended context for multi-step mathematical proofs
    - Length generalization for problems not seen during training
    - Mathematical notation handling
    """

    # Model architecture
    vocab_size: int = 50304 # Extended for mathematical symbols
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = None # For Grouped Query Attention (GQA)
    intermediate_size: int = 8192

    # Context and sequence lengths
    max_position_embeddings: int = 8192 # Extended context length
    max_sequence_length: int = 8192

    # Positional encoding
    rope_theta: float = 10000.0 # Base for Rotary Position Embeddings (RoPE)
    rope_scaling: Optional[dict] = None # For dynamic RoPE scaling
    use_dynamic_rope: bool = True # Enable dynamic RoPE for length generalization

    # Attention mechanisms
    attention_dropout: float = 0.0
    attention_bias: bool = False

    # Normalization
    rms_norm_eps: float = 1e-6

    # Regularization
    hidden_dropout: float = 0.1
    residual_dropout: float = 0.1

    # Activation
    hidden_act: str = "silu" # SiLU/Swish activation like Llama

    # Initialization
    initializer_range: float = 0.02

    # Training optimizations
    use_cache: bool = True
    gradient_checkpointing: bool = False

    # Mathematical reasoning specific
    use_flash_attention: bool = True # For efficiency with long sequences
    tie_word_embeddings: bool = False

    def __post_init__(self):
        """Validate and set dependent configurations."""
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        
        # Ensure intermediate size is divisible by hidden size
        if self.intermediate_size % self.hidden_size != 0:
            self.intermediate_size = ((self.intermediate_size // self.hidden_size) + 1) * self.hidden_size

        # Set up dynamic RoPE scaling if enabled
        if self.use_dynamic_rope and self.rope_scaling is None:
            self.rope_scaling = {
                "type": "dynamic",
                "factor": 2.0 # Allow 2x context extension
            }


def get_small_config() -> MathTransformerConfig:
    """Small configuration for testing and development."""
    return MathTransformerConfig(
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=2048,
        max_sequence_length=2048
    )

def get_base_config() -> MathTransformerConfig:
    """Base configuration (~1B parameters)."""
    return MathTransformerConfig(
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=8192,
        max_position_embeddings=8192,
        max_sequence_length=8192
    )

def get_large_config() -> MathTransformerConfig:
    """Large configuration (~7B parameters)."""
    return MathTransformerConfig(
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8, # GQA for efficiency
        intermediate_size=14336,
        max_position_embeddings=16384,
        max_sequence_length=16384
    )