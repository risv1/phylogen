"""
RoPE (Rotary Position Embedding) for DNA Embeddings

Implements Rotary Position Embeddings as described in
"RoFormer: Enhanced Transformer with Rotary Position Embedding"
https://arxiv.org/abs/2104.09864

Used in modern LLMs like LLaMA, GPT-NeoX, etc.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class RoPEEmbedder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 9,
        embed_dim: int = 256,
        max_len: int = 1024,
        dropout: float = 0.1,
        theta: float = 10000.0,
    ):
        """
        DNA Embedder with Rotary Position Embeddings (RoPE).

        Args:
            vocab_size: Number of tokens in vocabulary
            embed_dim: Dimension of embedding
            max_len: Maximum sequence length
            dropout: Dropout probability
            theta: Base for frequency computation (default 10000.0)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.theta = theta

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Precompute rotation frequencies
        # RoPE divides embedding into pairs and rotates each pair
        assert embed_dim % 2 == 0, "embed_dim must be even for RoPE"

        # Compute frequency for each dimension pair
        # freq_i = theta^(-2i/d) for i in [0, d/2)
        dim_pairs = embed_dim // 2
        freqs = 1.0 / (theta ** (torch.arange(0, embed_dim, 2).float() / embed_dim))

        # Precompute cos and sin for all positions up to max_len
        # Shape: (max_len, dim_pairs)
        positions = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        angles = positions * freqs.unsqueeze(0)  # (max_len, dim_pairs)

        # Cache cos and sin values
        # We'll need to apply these to interleaved dimensions
        self.register_buffer("cos_cached", torch.cos(angles))  # (max_len, dim_pairs)
        self.register_buffer("sin_cached", torch.sin(angles))  # (max_len, dim_pairs)

    def _apply_rotary_pos_emb(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply rotary position embeddings to input tensor.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            seq_len: Sequence length

        Returns:
            Tensor with rotary position embeddings applied
        """
        # Get cos and sin for current sequence length
        cos = self.get_buffer("cos_cached")[:seq_len, :]  # (seq_len, dim_pairs)
        sin = self.get_buffer("sin_cached")[:seq_len, :]  # (seq_len, dim_pairs)

        # Reshape x to separate even and odd dimensions
        # x shape: (batch, seq_len, embed_dim)
        # Split into (batch, seq_len, dim_pairs, 2)
        x_reshaped = x.reshape(x.shape[0], x.shape[1], -1, 2)

        # Extract even and odd dimensions
        x1 = x_reshaped[..., 0]  # (batch, seq_len, dim_pairs)
        x2 = x_reshaped[..., 1]  # (batch, seq_len, dim_pairs)

        # Apply rotation:
        # [x1'] = [cos  -sin] [x1]
        # [x2']   [sin   cos] [x2]
        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x1 * sin + x2 * cos

        # Stack back together
        x_rotated = torch.stack([x1_rotated, x2_rotated], dim=-1)

        # Reshape back to original shape
        x_rotated = x_rotated.reshape(x.shape[0], x.shape[1], self.embed_dim)

        return x_rotated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len)
        Returns:
            Tensor (batch, seq_len, embed_dim) with RoPE applied
        """
        batch_size, seq_len = x.size()

        # Get token embeddings and scale
        embedded = self.embedding(x) * math.sqrt(self.embed_dim)

        # Apply rotary position embeddings
        embedded = self._apply_rotary_pos_emb(embedded, seq_len)

        # Apply dropout
        embedded = self.dropout(embedded)

        return embedded


class RoPEEmbedderAlternative(nn.Module):
    """
    Alternative RoPE implementation using complex numbers.
    More efficient but mathematically equivalent.
    """

    def __init__(
        self,
        vocab_size: int = 9,
        embed_dim: int = 256,
        max_len: int = 1024,
        dropout: float = 0.1,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.theta = theta

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        assert embed_dim % 2 == 0, "embed_dim must be even for RoPE"

        # Compute frequencies
        dim_pairs = embed_dim // 2
        freqs = 1.0 / (theta ** (torch.arange(0, embed_dim, 2).float() / embed_dim))

        # Precompute rotation matrix as complex numbers
        positions = torch.arange(max_len).unsqueeze(1)
        angles = positions * freqs.unsqueeze(0)

        # e^(i*theta) = cos(theta) + i*sin(theta)
        freqs_complex = torch.polar(torch.ones_like(angles), angles)
        self.register_buffer("freqs_complex", freqs_complex)

    def _apply_rotary_pos_emb_complex(
        self, x: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """Apply RoPE using complex number multiplication."""
        # Get rotation matrix for current sequence
        freqs_cis = self.get_buffer("freqs_complex")[:seq_len, :]

        # Reshape x to (batch, seq_len, dim_pairs, 2)
        x_reshaped = x.float().reshape(x.shape[0], x.shape[1], -1, 2)

        # Convert to complex: x1 + i*x2
        x_complex = torch.view_as_complex(x_reshaped)

        # Multiply by rotation (broadcasting over batch dimension)
        x_rotated = x_complex * freqs_cis.unsqueeze(0)

        # Convert back to real
        x_out = torch.view_as_real(x_rotated)

        # Reshape to original
        x_out = x_out.reshape(x.shape[0], x.shape[1], self.embed_dim)

        return x_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len)
        Returns:
            Tensor (batch, seq_len, embed_dim) with RoPE applied
        """
        batch_size, seq_len = x.size()

        # Get token embeddings and scale
        embedded = self.embedding(x) * math.sqrt(self.embed_dim)

        # Apply rotary position embeddings
        embedded = self._apply_rotary_pos_emb_complex(embedded, seq_len)

        # Apply dropout
        embedded = self.dropout(embedded)

        return embedded
