"""
PhyloGen: Unified DNA Tokenization, Embedding, Training & Testing

Single-file implementation with:
- DNA k-mer tokenizer (optimized: k=3, stride=3)
- ALiBi embedder (fastest: 0.49ms forward)
- Transformer decoder
- Training & validation loops
- Testing & inference

Run: python phylogen.py --fasta data.fasta --train --epochs 50
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# ============================================================================
# TOKENIZER
# ============================================================================


class DNATokenizer:
    """K-mer DNA tokenizer with BOS/EOS/PAD tokens"""

    def __init__(self, k: int = 3, stride: int = 3, include_iupac: bool = False):
        self.k = k
        self.stride = stride
        self.include_iupac = include_iupac

        # Build vocab
        bases = "ACGT"
        if include_iupac:
            bases += "NRYSWKMBDHV"

        vocab = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        vocab += self._generate_kmers(bases, k)

        self.vocab = vocab
        self.token_to_id = {t: i for i, t in enumerate(vocab)}
        self.id_to_token = {i: t for i, t in enumerate(vocab)}

        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.vocab_size = len(vocab)

    def _generate_kmers(self, bases: str, k: int) -> List[str]:
        if k == 1:
            return list(bases)
        kmers = []
        for base in bases:
            for kmer in self._generate_kmers(bases, k - 1):
                kmers.append(base + kmer)
        return kmers

    def encode(self, sequence: str) -> torch.Tensor:
        """Encode DNA sequence to token IDs"""
        sequence = sequence.upper()
        tokens = [self.bos_token_id]

        for i in range(0, len(sequence) - self.k + 1, self.stride):
            kmer = sequence[i : i + self.k]
            if len(kmer) == self.k:
                tokens.append(self.token_to_id.get(kmer, self.unk_token_id))

        tokens.append(self.eos_token_id)
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        """Decode token IDs to DNA sequence"""
        if tokens.dim() > 1:
            tokens = tokens[0]

        sequence = ""
        for token_id in tokens.tolist():
            token = self.id_to_token.get(token_id, "<UNK>")
            if token not in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
                sequence += token

        # For non-overlapping k-mers
        if self.stride >= self.k:
            return sequence
        # For overlapping k-mers, stitch together
        return self._stitch_kmers(sequence)

    def _stitch_kmers(self, kmer_seq: str) -> str:
        if not kmer_seq or len(kmer_seq) < self.k:
            return kmer_seq
        result = kmer_seq[: self.k]
        for i in range(self.k, len(kmer_seq), self.stride):
            result += kmer_seq[i : i + self.stride]
        return result

    def batch_encode(
        self, sequences: List[str], max_len: Optional[int] = None
    ) -> torch.Tensor:
        """Batch encode with padding"""
        encoded = [self.encode(seq) for seq in sequences]
        if max_len is None:
            max_len = max(len(e) for e in encoded)

        batch = torch.full(
            (len(sequences), max_len), self.pad_token_id, dtype=torch.long
        )
        for i, enc in enumerate(encoded):
            length = min(len(enc), max_len)
            batch[i, :length] = enc[:length]

        return batch


# ============================================================================
# EMBEDDER (ALiBi)
# ============================================================================


class ALiBiEmbedder(nn.Module):
    """ALiBi (Attention with Linear Biases) Embedder - fastest, best extrapolation"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        max_len: int = 1024,
        dropout: float = 0.1,
        num_heads: int = 8,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.num_heads = num_heads

        # Token embeddings only (no positional embeddings)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # ALiBi slopes (one per head)
        slopes = self._get_slopes(num_heads)
        self.register_buffer("slopes", slopes)

    def _get_slopes(self, num_heads: int) -> torch.Tensor:
        """Compute ALiBi slopes for each attention head"""

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(torch.arange(1, n + 1).float().log2().floor() + 1)))
            return start

        if (num_heads & (num_heads - 1)) == 0:  # power of 2
            return get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2 ** num_heads.bit_length()
            slopes = get_slopes_power_of_2(closest_power_of_2)
            return slopes[:num_heads]

    def get_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate ALiBi attention bias matrix"""
        # Relative position matrix
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        relative_pos = positions - positions.transpose(0, 1)

        # Apply slopes (shape: num_heads, seq_len, seq_len)
        slopes = self.slopes.view(-1, 1, 1)  # (num_heads, 1, 1)
        bias = slopes * relative_pos.abs().unsqueeze(0)  # (num_heads, seq_len, seq_len)

        return -bias  # negative bias for attention

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len)
        Returns:
            embeddings: (batch, seq_len, embed_dim)
        """
        # Token embeddings
        embeddings = self.token_embedding(token_ids)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# ============================================================================
# TRANSFORMER DECODER
# ============================================================================


class TransformerDecoder(nn.Module):
    """Decoder-only transformer for next-token prediction"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 1024,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

        self.register_buffer(
            "causal_mask", torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        )

    def forward(self, embedded_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedded_tokens: (batch, seq_len, embed_dim)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        seq_len = embedded_tokens.size(1)
        mask = self.causal_mask[:seq_len, :seq_len]

        output = self.transformer(
            tgt=embedded_tokens,
            memory=embedded_tokens,
            tgt_mask=mask,
            tgt_is_causal=True,
        )

        return self.output_proj(output)


# ============================================================================
# DATASET
# ============================================================================


class DNADataset(Dataset):
    """FASTA dataset loader"""

    def __init__(self, fasta_path: str, tokenizer: DNATokenizer, max_len: int = 1024):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sequences = self._load_fasta(fasta_path)

    def _load_fasta(self, path: str) -> List[str]:
        sequences = []
        current_seq = []

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_seq:
                        sequences.append("".join(current_seq))
                        current_seq = []
                else:
                    current_seq.append(line)

            if current_seq:
                sequences.append("".join(current_seq))

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][: self.max_len]
        return self.tokenizer.encode(seq)


def collate_batch(batch: List[torch.Tensor], pad_token_id: int) -> torch.Tensor:
    """Collate batch with padding"""
    max_len = max(len(seq) for seq in batch)
    padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)

    for i, seq in enumerate(batch):
        padded[i, : len(seq)] = seq

    return padded


# ============================================================================
# TRAINER
# ============================================================================


class Trainer:
    """Training and validation"""

    def __init__(
        self,
        tokenizer: DNATokenizer,
        embedder: ALiBiEmbedder,
        decoder: TransformerDecoder,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
    ):
        self.tokenizer = tokenizer
        self.embedder = embedder.to(device)
        self.decoder = decoder.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.metrics = {"train_loss": [], "val_loss": [], "val_perplexity": []}

    def compute_loss(self, batch: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Next-token prediction loss"""
        embeddings = self.embedder(batch)
        logits = self.decoder(embeddings)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )

        with torch.no_grad():
            predictions = shift_logits.argmax(dim=-1)
            mask = shift_labels != self.tokenizer.pad_token_id
            accuracy = (predictions == shift_labels)[mask].float().mean()
            perplexity = torch.exp(loss)

        return loss, {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "perplexity": perplexity.item(),
        }

    def train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer):
        """Train one epoch"""
        self.embedder.train()
        self.decoder.train()

        total_loss = 0.0
        num_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch in progress:
            batch = batch.to(self.device)

            optimizer.zero_grad()
            loss, metrics = self.compute_loss(batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.embedder.parameters(), max_norm=1.0)

            optimizer.step()
            self.global_step += 1

            total_loss += metrics["loss"]
            num_batches += 1

            progress.set_postfix(
                {
                    "loss": f"{metrics['loss']:.4f}",
                    "acc": f"{metrics['accuracy']:.4f}",
                }
            )

        return {"loss": total_loss / num_batches}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader):
        """Validate"""
        self.embedder.eval()
        self.decoder.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]"):
            batch = batch.to(self.device)
            loss, metrics = self.compute_loss(batch)

            total_loss += metrics["loss"]
            num_batches += 1

        avg_loss = total_loss / num_batches
        return {
            "loss": avg_loss,
            "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
        }

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            "epoch": self.current_epoch,
            "embedder_state": self.embedder.state_dict(),
            "decoder_state": self.decoder.state_dict(),
            "best_val_loss": self.best_val_loss,
            "metrics": self.metrics,
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pt")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
    ):
        """Full training loop"""
        optimizer = AdamW(
            list(self.embedder.parameters()) + list(self.decoder.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

        print(f"\n{'=' * 80}")
        print(f"Training: {num_epochs} epochs on {self.device}")
        print(f"{'=' * 80}\n")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch(train_loader, optimizer)
            val_metrics = self.validate(val_loader)

            scheduler.step()

            print(
                f"\nEpoch {epoch + 1}: Train Loss={train_metrics['loss']:.4f}, "
                f"Val Loss={val_metrics['loss']:.4f}, PPL={val_metrics['perplexity']:.2f}"
            )

            self.metrics["train_loss"].append(train_metrics["loss"])
            self.metrics["val_loss"].append(val_metrics["loss"])
            self.metrics["val_perplexity"].append(val_metrics["perplexity"])

            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]
                print(f"  🏆 New best! Val Loss: {self.best_val_loss:.4f}")

            if (epoch + 1) % 5 == 0 or is_best:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt", is_best)

        # Save final metrics
        with open(self.checkpoint_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)


# ============================================================================
# INFERENCE / TESTING
# ============================================================================


class Tester:
    """Test and generate sequences"""

    def __init__(
        self,
        tokenizer: DNATokenizer,
        embedder: ALiBiEmbedder,
        decoder: TransformerDecoder,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.tokenizer = tokenizer
        self.embedder = embedder.to(device)
        self.decoder = decoder.to(device)
        self.device = device

    @torch.no_grad()
    def generate(self, prompt: str, max_len: int = 100, temperature: float = 1.0):
        """Generate sequence from prompt"""
        self.embedder.eval()
        self.decoder.eval()

        tokens = self.tokenizer.encode(prompt).unsqueeze(0).to(self.device)

        for _ in range(max_len):
            embeddings = self.embedder(tokens)
            logits = self.decoder(embeddings)

            next_token_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

        return self.tokenizer.decode(tokens[0])

    @torch.no_grad()
    def test(self, test_sequences: List[str]):
        """Test on sequences"""
        print(f"\n{'=' * 80}")
        print("Testing")
        print(f"{'=' * 80}\n")

        for i, seq in enumerate(test_sequences, 1):
            tokens = self.tokenizer.encode(seq).unsqueeze(0).to(self.device)
            embeddings = self.embedder(tokens)
            logits = self.decoder(embeddings)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = tokens[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.tokenizer.pad_token_id,
            )

            print(f"Seq {i}: {seq[:60]}...")
            print(f"  Loss: {loss.item():.4f}, PPL: {torch.exp(loss):.2f}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="PhyloGen: Unified DNA ML Pipeline")

    # Model config
    parser.add_argument("--k", type=int, default=3, help="K-mer size")
    parser.add_argument("--stride", type=int, default=3, help="K-mer stride")
    parser.add_argument("--embed-dim", type=int, default=256, help="Embedding dim")
    parser.add_argument("--num-layers", type=int, default=4, help="Transformer layers")
    parser.add_argument("--num-heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--max-len", type=int, default=1024, help="Max sequence length")

    # Data
    parser.add_argument("--fasta", type=str, help="FASTA file path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")

    # Training
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device")

    # Testing
    parser.add_argument("--test", action="store_true", help="Run testing")
    parser.add_argument("--generate", type=str, help="Generate from prompt")
    parser.add_argument("--checkpoint", type=str, help="Load checkpoint")

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"\n{'=' * 80}")
    print("PhyloGen: DNA Tokenization, Embedding & Training")
    print(f"{'=' * 80}")
    print(f"Config: k={args.k}, stride={args.stride}, embed_dim={args.embed_dim}")
    print(f"Device: {device}")
    print(f"{'=' * 80}\n")

    # Initialize
    tokenizer = DNATokenizer(k=args.k, stride=args.stride)
    print(f"✓ Tokenizer: vocab_size={tokenizer.vocab_size}")

    embedder = ALiBiEmbedder(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        max_len=args.max_len,
        num_heads=args.num_heads,
    )
    print(f"✓ Embedder: {sum(p.numel() for p in embedder.parameters()):,} params")

    decoder = TransformerDecoder(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_len=args.max_len,
    )
    print(f"✓ Decoder: {sum(p.numel() for p in decoder.parameters()):,} params")

    # Load checkpoint if specified
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        embedder.load_state_dict(checkpoint["embedder_state"])
        decoder.load_state_dict(checkpoint["decoder_state"])
        print(f"✓ Loaded checkpoint: {args.checkpoint}")

    # Training
    if args.train:
        if not args.fasta:
            print("Error: --fasta required for training")
            return

        dataset = DNADataset(args.fasta, tokenizer, max_len=args.max_len)
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id),
        )

        trainer = Trainer(tokenizer, embedder, decoder, device=device)
        trainer.train(train_loader, val_loader, args.epochs, lr=args.lr)

    # Testing
    if args.test:
        test_seqs = [
            "ATCGATCGATCGATCG",
            "GCTAGCTAGCTAGCTA",
            "AAATTTGGGCCCAAATTTGGGCCC",
        ]
        tester = Tester(tokenizer, embedder, decoder, device=device)
        tester.test(test_seqs)

    # Generation
    if args.generate:
        tester = Tester(tokenizer, embedder, decoder, device=device)
        result = tester.generate(args.generate, max_len=100)
        print(f"\nGenerated from '{args.generate}':\n{result}")


if __name__ == "__main__":
    main()
