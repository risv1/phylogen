"""
Data loading utilities for PhyloGen

Contains utilities for loading and preprocessing DNA sequences from FASTA files,
including the CARD (Comprehensive Antibiotic Resistance Database) dataset.
"""

from typing import List, Optional

import torch
from Bio import SeqIO
from torch.utils.data import Dataset


class DNADataset(Dataset):
    """
    PyTorch Dataset for loading DNA sequences from FASTA files.
    
    Supports any FASTA file including CARD database sequences.
    """
    
    def __init__(
        self, 
        fasta_file: str, 
        tokenizer, 
        max_len: int = 1024,
        filter_invalid: bool = True
    ):
        """
        Dataset for loading DNA sequences from a FASTA file.

        Args:
            fasta_file: Path to the FASTA file (e.g., CARD database)
            tokenizer: Instance of tokenizer (DNATokenizer or BPETokenizer)
            max_len: Maximum sequence length to truncate to (including specials)
            filter_invalid: Whether to filter sequences with invalid characters
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sequences = []
        self.headers = []

        # Load sequences
        print(f"Loading sequences from {fasta_file}...")
        skipped = 0
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq_str = str(record.seq).upper()
            
            # Filter invalid sequences if requested
            if filter_invalid and not all(c in "ACGT" for c in seq_str):
                skipped += 1
                continue
                
            if len(seq_str) > 0:  # Skip empty sequences
                self.sequences.append(seq_str)
                self.headers.append(record.id)
                
        print(f"Loaded {len(self.sequences)} sequences.")
        if skipped > 0:
            print(f"Skipped {skipped} sequences with invalid characters.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a tokenized sequence by index.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tokenized sequence tensor
        """
        seq = self.sequences[idx]
        
        # Tokenize (encode adds BOS/EOS)
        token_ids = self.tokenizer.encode(seq, validate=False)

        # Truncate if necessary
        if len(token_ids) > self.max_len:
            # Keep BOS, truncate middle, keep EOS
            bos_token = token_ids[0]
            eos_token = token_ids[-1]
            middle_len = self.max_len - 2
            token_ids = torch.cat([
                token_ids[:1],  # BOS
                token_ids[1:middle_len+1],  # Middle part
                token_ids[-1:]  # EOS
            ])

        return token_ids
    
    def get_sequence(self, idx: int) -> str:
        """Get the raw DNA sequence string by index."""
        return self.sequences[idx]
    
    def get_header(self, idx: int) -> str:
        """Get the FASTA header/ID by index."""
        return self.headers[idx]


def collate_dna_batch(batch: List[torch.Tensor], pad_token_id: int) -> torch.Tensor:
    """
    Collate function to pad batches of sequences to the same length.
    
    Args:
        batch: List of tokenized sequences
        pad_token_id: Token ID to use for padding
        
    Returns:
        Padded batch tensor of shape (batch_size, max_seq_len)
    """
    from torch.nn.utils.rnn import pad_sequence

    # pad_sequence expects a list of tensors (L_i)
    # batch_first=True -> (B, L_max)
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=pad_token_id)
    return padded_batch


def load_fasta_sequences(
    fasta_file: str, 
    max_sequences: Optional[int] = None,
    filter_invalid: bool = True
) -> List[str]:
    """
    Load DNA sequences from a FASTA file without tokenization.
    
    Useful for benchmarking and quick data exploration.
    
    Args:
        fasta_file: Path to FASTA file
        max_sequences: Maximum number of sequences to load (None for all)
        filter_invalid: Whether to filter sequences with non-ACGT characters
        
    Returns:
        List of DNA sequence strings
    """
    sequences = []
    
    with open(fasta_file, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            seq_str = str(record.seq).upper()
            
            # Filter invalid sequences if requested
            if filter_invalid and not all(c in "ACGT" for c in seq_str):
                continue
                
            if len(seq_str) > 0:
                sequences.append(seq_str)
                
            if max_sequences and len(sequences) >= max_sequences:
                break
                
    return sequences
