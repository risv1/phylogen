"""
ALiBi (Attention with Linear Biases) Embedder for DNA Sequences

Implements ALiBi as described in "Train Short, Test Long: Attention with Linear Biases
Enables Input Length Extrapolation" (Press et al., 2022)
https://arxiv.org/abs/2108.12409

ALiBi does not use positional embeddings. Instead, it adds a linear bias
to attention scores based on distance between positions.
"""

import math

import torch
import torch.nn as nn


class ALiBiEmbedder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 9,
        embed_dim: int = 256,
        max_len: int = 1024,
        dropout: float = 0.1,
        num_heads: int = 8,
    ):
        """
        DNA Embedder with ALiBi (no positional encodings).

        Args:
            vocab_size: Number of tokens in vocabulary
            embed_dim: Dimension of embedding
            max_len: Maximum sequence length (for pre-computing biases)
            dropout: Dropout probability
            num_heads: Number of attention heads (for computing slopes)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_len = max_len

        # Token embedding only (no positional encoding)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Compute ALiBi slopes for each attention head
        # Slopes are geometric sequence: m_i = 2^(-8i/n) where n is num_heads
        slopes = self._get_alibi_slopes(num_heads)
        self.register_buffer("slopes", slopes)

        # Precompute bias matrix for efficiency
        # Shape: (num_heads, max_len, max_len)
        bias_matrix = self._build_alibi_bias_matrix(max_len, slopes)
        self.register_buffer("bias_matrix", bias_matrix)

    def _get_alibi_slopes(self, num_heads: int) -> torch.Tensor:
        """
        Compute ALiBi slopes for attention heads.

        For num_heads = 8:
        slopes = [2^-1, 2^-2, 2^-3, 2^-4, 2^-5, 2^-6, 2^-7, 2^-8]
               = [0.5, 0.25, 0.125, 0.0625, ...]

        Args:
            num_heads: Number of attention heads

        Returns:
            Tensor of slopes, shape (num_heads,)
        """

        # Compute the base slope factor
        # For 8 heads: 2^(-8/8) = 2^-1 = 0.5
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio**i) for i in range(n)]

        # For non-power-of-2 heads, use closest power of 2 and interpolate
        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            # Use closest power of 2
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)

            # Add extra slopes by interpolation
            extra_slopes_count = num_heads - closest_power_of_2
            extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
            extra_slopes = [
                extra_base * (extra_base**i) for i in range(extra_slopes_count)
            ]
            slopes = slopes + extra_slopes

        return torch.tensor(slopes, dtype=torch.float32)

    def _build_alibi_bias_matrix(
        self, max_len: int, slopes: torch.Tensor
    ) -> torch.Tensor:
        """
        Build the ALiBi bias matrix for all heads.

        The bias for position i attending to position j is:
        bias[i, j] = -slope * |i - j|

        Args:
            max_len: Maximum sequence length
            slopes: Slopes for each head, shape (num_heads,)

        Returns:
            Bias matrix of shape (num_heads, max_len, max_len)
        """
        # Create position distance matrix
        # distances[i, j] = |i - j|
        positions = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        distances = torch.abs(positions - positions.T)  # (max_len, max_len)

        # Apply slopes to get biases for each head
        # Shape: (num_heads, max_len, max_len)
        biases = -slopes.unsqueeze(1).unsqueeze(2) * distances.unsqueeze(0)

        return biases

    def get_alibi_bias(self, seq_len: int) -> torch.Tensor:
        """
        Get ALiBi bias matrix for current sequence length.

        This can be added to attention scores in the transformer.

        Args:
            seq_len: Current sequence length

        Returns:
            Bias matrix of shape (num_heads, seq_len, seq_len)
        """
        if seq_len <= self.max_len:
            # Use precomputed bias
            return self.bias_matrix[:, :seq_len, :seq_len]
        else:
            # Recompute for longer sequences (extrapolation)
            slopes = self.slopes
            return self._build_alibi_bias_matrix(seq_len, slopes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len)
        Returns:
            Tensor (batch, seq_len, embed_dim) WITHOUT positional encodings.
            ALiBi biases are added in the attention layer, not here.
        """
        batch_size, seq_len = x.size()

        # Get token embeddings and scale
        # Note: No positional encoding added here!
        embedded = self.embedding(x) * math.sqrt(self.embed_dim)

        # Apply dropout
        embedded = self.dropout(embedded)

        return embedded


class ALiBiEmbedderSimple(nn.Module):
    """
    Simplified ALiBi embedder that just returns token embeddings.
    The bias computation is deferred to the attention mechanism.
    """

    def __init__(
        self,
        vocab_size: int = 9,
        embed_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Simple DNA Embedder for ALiBi (no positional information).

        Args:
            vocab_size: Number of tokens in vocabulary
            embed_dim: Dimension of embedding
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len)
        Returns:
            Tensor (batch, seq_len, embed_dim) with only token embeddings
        """
        # Token embeddings with scaling
        embedded = self.embedding(x) * math.sqrt(self.embed_dim)

        # Apply dropout
        embedded = self.dropout(embedded)

        return embedded
