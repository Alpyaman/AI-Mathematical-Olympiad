"""
Decoder-only Transformer Architecture for Mathematical Reasoning
Based on Llama/GPT architecture with customizations for math problems
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass

from .rope import RotaryPositionalEmbedding, DynamicRoPE

@dataclass
class CacheState:
    """Cache for key and value tensors during inference."""
    key: torch.Tensor
    value: torch.Tensor
    seq_len: int


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    More efficient than LayerNorm and shown to work well in LLMs.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with optional Grouped-Query Attention (GQA).

    Features:
    - Grouped-Query Attention for efficiency
    - RoPE positional encoding
    - KV caching for inference
    - Optional Flash Attention
    """

    def __init__(self, hidden_size: int, num_heads: int, num_key_value_heads: Optional[int] = None, dropout: float = 0.0, bias: bool = False, rope: Optional[nn.Module] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.rope = rope

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value tensors for grouped-query attention."""
        if n_rep == 1:
            return hidden_states

        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[CacheState] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[CacheState]]:
        """
        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size)
            attention_mask: Mask tensor (batch_size, 1, seq_len, seq_len)
            position_ids: Position indices (batch_size, seq_len)
            past_key_value: Cached key/value from previous steps
            use_cache: Whether to return cache for next step

        Returns:
            Tuple of (output, new_cache)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if self.rope is not None:
            query, key = self.rope(query, key, position_ids)

        # Handle KV cache
        if past_key_value is not None:
            key = torch.cat([past_key_value.key, key], dim=2)
            value = torch.cat([past_key_value.value, value], dim=2)

        kv_seq_len = key.shape[2]
        new_cache = CacheState(key=key, value=value, seq_len=kv_seq_len) if use_cache else None

        # Repeat K,V for grouped-query attention
        key = self._repeat_kv(key, self.num_key_value_groups)
        value = self._repeat_kv(value, self.num_key_value_groups)

        # Compute attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, new_cache


class MLP(nn.Module):
    """
    Feed-forward network with SiLU activation (SwiGLU variant possible).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Activation function
        if hidden_act == "silu":
            self.act_fn = F.silu
        elif hidden_act == "gelu":
            self.act_fn = F.gelu
        elif hidden_act == "relu":
            self.act_fn = F.relu
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU-style feed-forward network."""
        # gate = self.act_fn(self.gate_proj(x))
        # up = self.up_proj(x)
        # down = self.down_proj(gate * up)
        gate = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        down = self.down_proj(gate)
        return self.dropout(down)


class DecoderLayer(nn.Module):
    """
    Single transformer decoder layer.

    Architecture:
    - Pre-norm with RMSNorm
    - Multi-head attention with RoPE
    - Feed-forward network
    - Residual connections
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        attention_bias: bool = False,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            dropout=attention_dropout,
            bias=attention_bias,
            rope=rope,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            dropout=hidden_dropout,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[CacheState] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[CacheState]]:
        """
        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached KV from previous step
            use_cache: Whether to cache KV

        Returns:
            Tuple of (output, cache)
        """
        residual = hidden_states

        # Self-attention with pre-norm
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Feed-forward with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_cache


class MathTransformerDecoder(nn.Module):
    """
    Decoder-only Transformer for Mathematical Reasoning.

    Features:
    - Llama-style architecture with RMSNorm and SwiGLU
    - RoPE positional encoding with dynamic scaling
    - Grouped-Query Attention for efficiency
    - Extended context length for long proofs
    - KV caching for efficient inference
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Positional encoding
        if config.use_dynamic_rope:
            self.rope = DynamicRoPE(
                dim=config.hidden_size // config.num_attention_heads,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
        else:
            self.rope = RotaryPositionalEmbedding(
                dim=config.hidden_size // config.num_attention_heads,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                rms_norm_eps=config.rms_norm_eps,
                attention_dropout=config.attention_dropout,
                hidden_dropout=config.hidden_dropout,
                attention_bias=config.attention_bias,
                rope=self.rope,
            )
            for _ in range(config.num_hidden_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM head (output projection)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embeddings and output weights if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Gradient checkpointing
        self.gradient_checkpointing = config.gradient_checkpointing

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following standard practice."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def _prepare_decoder_attention_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Create causal attention mask.

        Args:
            attention_mask: Padding mask (batch_size, seq_len)
            input_shape: Shape of input (batch_size, seq_len)
            dtype: Data type for mask

        Returns:
            Combined causal and padding mask
        """
        batch_size, seq_len = input_shape

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=attention_mask.device),
            diagonal=1
        )
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)

        # Combine with padding mask
        if attention_mask is not None:
            padding_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            padding_mask = padding_mask.expand(batch_size, 1, seq_len, seq_len)
            combined_mask = causal_mask | ~padding_mask
        else:
            combined_mask = causal_mask

        # Convert to attention scores mask
        mask = torch.zeros_like(combined_mask, dtype=dtype)
        mask.masked_fill_(combined_mask, float("-inf"))

        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token ids (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            position_ids: Position indices (batch_size, seq_len)
            past_key_values: Cached KV states for each layer
            use_cache: Whether to cache KV states
            labels: Labels for language modeling loss (batch_size, seq_len)

        Returns:
            Dictionary with logits, loss, and optionally cached states
        """
        batch_size, seq_len = input_ids.shape

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)

        # Create position ids if not provided
        if position_ids is None:
            past_length = 0
            # If we have a cache, the current position starts after the cached sequence
            if past_key_values is not None and past_key_values[0] is not None:
                past_length = past_key_values[0].seq_len
            
            position_ids = torch.arange(past_length, past_length + seq_len, device=input_ids.device).unsqueeze(0)

        # Prepare attention mask
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_len), self.embed_tokens.weight.dtype
        )

        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Initialize cache if needed
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        # Pass through decoder layers
        new_key_values = [] if use_cache else None
        for layer_idx, (layer, past_kv) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, new_kv = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            if use_cache:
                new_key_values.append(new_kv)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Project to vocabulary
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "logits": logits,
            "loss": loss,
            "past_key_values": new_key_values,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,  # <--- NEW ARGUMENT
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        """
        self.eval()
        past_key_values = None

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                logits = outputs["logits"][:, -1, :]
                past_key_values = outputs["past_key_values"]

                # Apply temperature
                if temperature > 0:
                    logits = logits / temperature
                else:
                    do_sample = False # Greedy if temp is 0

                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")

                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                # Sample or take argmax
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                # Check for EOS token
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
                # Fallback: Stop if we hit 0 (pad) only if EOS wasn't specified
                elif eos_token_id is None and next_token.item() == 0:
                    break

        return input_ids