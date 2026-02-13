import torch.nn as nn

from attention import PhyloAttention

class PhyloGenBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        ff_dim = ff_dim or embed_dim * 4

        self.attn = PhyloAttention(embed_dim, num_heads, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, phylo_dists, alibi_bias, attn_mask):
        # Attention path
        attn_out = self.attn(self.norm1(x), phylo_dists, alibi_bias, attn_mask)
        x = x + attn_out

        # Feed-forward path
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out

        return x
