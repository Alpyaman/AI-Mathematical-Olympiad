"""
AI Mathematical Olympiad - Phase 1.1
Decoder-only Transformer Architecture for Mathematical Reasoning
"""

__version__ = "0.1.0"

# Always available: tokenizer (no torch dependency)
from .tokenizer.math_tokenizer import MathTokenizer

# Configuration (no torch dependency)
from .config.model_config import (
    MathTransformerConfig,
    get_small_config,
    get_base_config,
    get_large_config,
)

# Model components (require torch)
try:
    from .model.decoder import MathTransformerDecoder
    from .model.rope import RotaryPositionalEmbedding, DynamicRoPE

    __all__ = [
        "MathTransformerConfig",
        "MathTransformerDecoder",
        "MathTokenizer",
        "RotaryPositionalEmbedding",
        "DynamicRoPE",
        "get_small_config",
        "get_base_config",
        "get_large_config",
    ]
except ImportError as e:
    # Torch not available, only export tokenizer and config
    __all__ = [
        "MathTransformerConfig",
        "MathTokenizer",
        "get_small_config",
        "get_base_config",
        "get_large_config",
    ]