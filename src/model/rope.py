"""
Rotary Position Embedding (RoPE) Implementation
With support for dynamic scaling and length generalization.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) with dynamic scaling support.

    RoPE applies rotations to query and key embeddings based on their positions,
    enabling better length generalization compared to absolute positional encodings.

    Features:
    - Dynamic scaling for extrapolation to longer sequences.
    - Efficient caching for inference.
    - Support for grouped-query attention (GQA).
    """

    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0, scaling_factor: float = 1.0, device: Optional[torch.device] = None):
        """
        Args:
            dim: Dimension of each attention head.
            max_position_embeddings: Maximum sequence length.
            base: Base for computing rotation frequencies.
            scaling_factor: Scaling factor for dynamic RoPE scaling.
            device: Device to place the tensors on.
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor


        # Compute inverse frequencies for rotations
        inv_freq = 1.0 / self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache for cos and sin values
        self._cos_cache = None
        self._sin_cache = None
        self._seq_len_cached = 0
    
    def _compute_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Compute and cache cos and sin values for given sequence length."""
        if seq_len > self._seq_len_cached or self._cos_cache is None:
            self._seq_len_cached = max(seq_len, self._seq_len_cached)

            # Create position indices
            t = torch.arange(self._seq_len_cached, device=device, dtype=self.inv_freq.dtype)

            # Apply scaling for dynamic RoPE
            if self.scaling_factor != 1.0:
                t = t / self.scaling_factor
            
            # Compute frequencies
            freqs = torch.outers(t, self.inv_freq)

            # Create rotation matrix components
            emb = torch.cat([freqs, freqs], dim=-1)

            self._cos_cache = emb.cos().to(dtype)
            self._sin_cache = emb.sin().to(dtype)
        
        return self._cos_cache[:seq_len], self._sin_cache[:seq_len]
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dimensions of the input."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            k: Key tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
            position_ids: Optional position indices (batch_size, seq_len)
        
        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Get cos and sin values
        cos, sin = self._compute_cos_sin(seq_len, q.device, q.dtype)

        # Handle custom position_ids if provided
        if position_ids is not None:
            cos = cos[position_ids].unsqueeze(1)  # (batch_size, 1, seq_len, head_dim)
            sin = sin[position_ids].unsqueeze(1)  # (batch_size, 1, seq_len, head_dim)
        else:
            cos = cos[None, None, :, :] # (1, 1, seq_len, dim)
            sin = sin[None, None, :, :]

        # Apply rotation
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed
    
    def reset_cache(self):
        """Reset cached cos/sin values."""
        self._cos_cache = None
        self._sin_cache = None
        self._seq_len_cached = 0


class DynamicRoPE(RotaryPositionEmbedding):
    """
    Dynammic RoPE with automatic scaling factor adjustment.

    This variant automatically adjusts the scaling factor based on the sequence length to
    enable better extrapolation to longer sequences.
    """

    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0, max_scaling_factor: float = 4.0, device: Optional[torch.devices] = None):
        super().__init__(dim, max_position_embeddings, base, 1.0, device)
        self.max_scaling_factor = max_scaling_factor

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply dynamic RoPE with automatic scaling."""
        seq_len = q.shape[2]

        # Adjust scaling factor based on sequence length
        if seq_len > self.max_position_embeddings:
            self.scaling_factor = min(seq_len / self.max_position_embeddings, self.max_scaling_factor)
        else:
            self.scaling_factor = 1.0
        
        return super().forward(q, k, position_ids)