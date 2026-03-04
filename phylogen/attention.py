import math
from typing import Optional

import torch
import torch.nn as nn

class PhyloAttention(nn.Module):
    """
    Multi-head attention with:
    - ALiBi linear bias (no learned positional embeddings)
    - Phylogenetic distance bias (learned scalar weight)
    - Causal masking
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Learned strength of phylogenetic bias
        self.phylo_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x: torch.Tensor,                    # (B, L, D)
        phylo_dists: torch.Tensor | None,   # Allow None
        alibi_bias: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        B, L, D = x.shape

        qkv = self.qkv_proj(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        scores = (q @ k.transpose(-2, -1)) * self.scale

        scores = scores + alibi_bias.unsqueeze(0)

        # Phylo bias — skip if None
        if phylo_dists is not None:
            if phylo_dists.dim() == 2:  # (B, num_reps)
                phylo_scalar = phylo_dists.mean(dim=-1, keepdim=True)
                phylo_bias = phylo_scalar.unsqueeze(1).unsqueeze(2)
            else:
                phylo_bias = phylo_dists
            scores = scores + self.phylo_alpha * phylo_bias

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)