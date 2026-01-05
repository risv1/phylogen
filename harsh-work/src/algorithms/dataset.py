from typing import List

import torch
from Bio import SeqIO
from torch.utils.data import Dataset

from algorithms.tokenizer import DNATokenizer


class DNADataset(Dataset):
    def __init__(self, fasta_file: str, tokenizer: DNATokenizer, max_len: int = 1024):
        """
        Dataset for loading DNA sequences from a FASTA file.

        Args:
            fasta_file: Path to the FASTA file.
            tokenizer: Instance of DNATokenizer.
            max_len: Maximum sequence length to truncate to (including specials).
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sequences = []

        # Load sequences
        print(f"Loading sequences from {fasta_file}...")
        for record in SeqIO.parse(fasta_file, "fasta"):
            self.sequences.append(str(record.seq))
        print(f"Loaded {len(self.sequences)} sequences.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.sequences[idx]
        # Tokenize (encode adds BOS/EOS)
        token_ids = self.tokenizer.encode(seq)

        # Truncate if necessary (keeping BOS and EOS is tricky if we just slice)
        # Strategy: Slice the middle, keep BOS/EOS if possible, or just slice strictly.
        # Simple truncation for now:
        if len(token_ids) > self.max_len:
            token_ids = token_ids[: self.max_len]
            # Ensure EOS is present if we truncated?
            # For autoregressive training, it's often better to just slice.
            # But let's try to preserve EOS if we can, or just accept truncation.
            # Let's just slice for simplicity in this prototype.

        return token_ids

        # Let's just slice for simplicity in this prototype.

        return token_ids


def collate_dna_batch(batch: List[torch.Tensor], pad_token_id: int) -> torch.Tensor:
    """
    Collate function to pad batches of sequences.
    """
    from torch.nn.utils.rnn import pad_sequence

    # pad_sequence expects a list of tensors (L_i)
    # batch_first=True -> (B, L_max)
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=pad_token_id)
    return padded_batch
