"""
AI Mathematical Olympiad -Phase 1.1
Decoder-only Transformer Architecture for Mathematical Reasoning
"""

from .config.model_config import (
    MathTransformerConfig,
    get_small_config,
    get_base_config,
    get_large_config,
)

from .model.decoder import MathTransformerDecoder
from .model.rope import RotaryPositionalEmbedding, DynamicRoPE
from .tokenizer.math_tokenizer import MathTokenizer

__version__ = "0.1.0"

__all__ = [
    "MathTransformerConfig",
    "MathTransformerDecoder",
    "RotaryPositionalEmbedding",
    "DynamicRoPE",
    "MathTokenizer",
    "get_small_config",
    "get_base_config",
    "get_large_config",
]