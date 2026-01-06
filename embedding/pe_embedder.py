import math

import torch
import torch.nn as nn


class PEEmbedder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 9,
        embed_dim: int = 256,
        max_len: int = 1024,
        pos_type: str = "sinusoidal",
        dropout: float = 0.1,
    ):
        """
        DNA Embedder with configurable Positional Encoding.

        Args:
            vocab_size: Number of tokens in vocabulary.
            embed_dim: Dimension of embedding.
            max_len: Maximum sequence length.
            pos_type: Type of positional encoding ('sinusoidal' or 'learnable')
            dropout: Dropout probability for embeddings
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_type = pos_type

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Create positional encodings based on type
        if pos_type == "sinusoidal":
            # Fixed sinusoidal encodings (not learnable)
            self.register_buffer(
                "pos_encoding", self._create_sinusoidal_encoding(max_len, embed_dim)
            )
        elif pos_type == "learnable":
            # Learnable positional embeddings
            self.pos_encoding = nn.Embedding(max_len, embed_dim)
        else:
            raise ValueError(
                f"pos_type must be 'sinusoidal' or 'learnable', got {pos_type}"
            )

    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create fixed sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len)
        Returns:
            Tensor (batch, seq_len, embed_dim) with positional encodings added.
        """
        batch_size, seq_len = x.size()

        # Scale embeddings by sqrt(d_model) as per Attention Is All You Need
        embedded = self.embedding(x) * math.sqrt(self.embed_dim)

        # Add positional encoding based on type
        if self.pos_type == "sinusoidal":
            # Add fixed sinusoidal encodings (sliced to current sequence length)
            # Access registered buffer using get_buffer
            pos_enc = self.get_buffer("pos_encoding")
            embedded = embedded + pos_enc[:, :seq_len, :]
        elif self.pos_type == "learnable":
            # Add learnable positional embeddings
            positions = (
                torch.arange(seq_len, device=x.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            embedded = embedded + self.pos_encoding(positions)

        # Apply dropout for regularization
        embedded = self.dropout(embedded)

        return embedded
