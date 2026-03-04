import sys
import math
from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent.parent))

from .attention import PhyloAttention
from .block import PhyloGenBlock

from embedding.alibi_embedder import ALiBiEmbedder

class PhyloGen(nn.Module):
    """
    Decoder-only transformer with:
    - Token embedding
    - ALiBi positional bias
    - Phylogenetic attention bias
    - Standard transformer blocks
    """
    def __init__(
        self,
        vocab_size: int,
        tokenizer=None, # pass tokenizer for dynamic pad id
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        if tokenizer is not None:
            self.pad_token_id = tokenizer.pad_token_id
        else:
            self.pad_token_id = 31  # fallback

        # Token embedding
        self.token_embed = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=self.pad_token_id
        )
        self.embed_dropout = nn.Dropout(dropout)

        # ALiBi component
        self.alibi = ALiBiEmbedder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_len=max_seq_len,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Stack of transformer blocks (these are the decoder layers)
        self.blocks = nn.ModuleList([
            PhyloGenBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final norm + LM head
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02 / math.sqrt(self.embed_dim))

    def forward(
        self,
        input_ids: torch.Tensor,
        phylo_dists: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        sep_pos: Optional[torch.Tensor] = None,  # new: (B,)
        return_dict: bool = True,
    ):
        B, L = input_ids.shape

        x = self.token_embed(input_ids) * math.sqrt(self.embed_dim)
        x = self.embed_dropout(x)

        alibi_bias = self.alibi.get_alibi_bias(L)
        attn_mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool)).view(1, 1, L, L)

        for block in self.blocks:
            x = block(x, phylo_dists, alibi_bias, attn_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Mask prefix up to separator (finetune mode)
            if sep_pos is not None:
                ignore_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
                for b in range(B):
                    pos = sep_pos[b].item()
                    if pos >= 0:
                        ignore_mask[b, :pos] = True  # ignore before SEP
                shift_labels = shift_labels.masked_fill(ignore_mask, -100)

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        if return_dict:
            return {"logits": logits, "loss": loss}
        return logits

    def to(self, *args, **kwargs):
        """
        Override .to() so ALiBi buffers move to the correct device too
        """
        super().to(*args, **kwargs)
        self.alibi.to(*args, **kwargs)  # important for precomputed biases
        return self


# ────────────────────────────────────────────────
# Quick test / usage example
# ────────────────────────────────────────────────

if __name__ == "__main__":
    from tokenizer import ProteinTokenizer

    tokenizer_path = Path(__file__).resolve().parent.parent / "tokenizer" / "tokenizer.json"
    tokenizer = ProteinTokenizer.load(str(tokenizer_path))

    model = PhyloGen(
        vocab_size=tokenizer.vocab_size,
        tokenizer=tokenizer, # now dynamic pad id
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        max_seq_len=2048,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    dummy_ids = torch.randint(0, tokenizer.vocab_size, (2, 128), device=device)
    dummy_phylo = torch.rand(2, 20, device=device) * 0.05

    out = model(dummy_ids, dummy_phylo, labels=dummy_ids)
    print("Device:", next(model.parameters()).device)
    print("Logits shape:", out["logits"].shape)
    print("Loss:", out["loss"].item() if out["loss"] is not None else "None")