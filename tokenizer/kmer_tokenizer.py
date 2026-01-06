import json
from typing import Dict, List, Optional

import torch


class KmerTokenizer:
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        k: int = 1,
        stride: int = 1,
        include_iupac: bool = False,
    ):
        """
        Tokenizer with support for character-level and k-mer tokenization.

        Args:
            vocab: Custom vocabulary mapping. If None, generates based on k and include_iupac.
            k: K-mer size (1 for character-level, 3 for codons, etc.)
            stride: Stride for k-mer extraction (1 for overlapping, k for non-overlapping)
            include_iupac: Include IUPAC ambiguous nucleotide codes
        """
        self.k = k
        self.stride = stride
        self.include_iupac = include_iupac

        if vocab is None:
            vocab = self._generate_vocab(k, include_iupac)

        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = self.vocab["[PAD]"]
        self.bos_token_id = self.vocab["[BOS]"]
        self.eos_token_id = self.vocab["[EOS]"]
        self.unk_token_id = self.vocab["[UNK]"]

        # IUPAC nucleotide codes mapping
        self.iupac_codes = {
            "A": "A",
            "C": "C",
            "G": "G",
            "T": "T",
            "R": "AG",
            "Y": "CT",
            "S": "GC",
            "W": "AT",
            "K": "GT",
            "M": "AC",
            "B": "CGT",
            "D": "AGT",
            "H": "ACT",
            "V": "ACG",
            "N": "ACGT",
        }

    def _generate_vocab(self, k: int, include_iupac: bool) -> Dict[str, int]:
        """Generate vocabulary for k-mers."""
        vocab = {}
        idx = 0

        # Generate all k-mers
        bases = ["A", "C", "G", "T"]
        if k == 1:
            for base in bases:
                vocab[base] = idx
                idx += 1
        else:
            # Generate all k-mer combinations
            from itertools import product

            for kmer in product(bases, repeat=k):
                vocab["".join(kmer)] = idx
                idx += 1

        # Add IUPAC codes if requested (only for k=1)
        if include_iupac and k == 1:
            for code in ["R", "Y", "S", "W", "K", "M", "B", "D", "H", "V", "N"]:
                vocab[code] = idx
                idx += 1

        # Add special tokens
        vocab["[BOS]"] = idx
        idx += 1
        vocab["[EOS]"] = idx
        idx += 1
        vocab["[PAD]"] = idx
        idx += 1
        vocab["[UNK]"] = idx
        idx += 1
        vocab["[MUT]"] = idx

        return vocab

    def validate_sequence(self, sequence: str) -> bool:
        """
        Validate if sequence contains only valid nucleotide codes.

        Args:
            sequence: DNA sequence string

        Returns:
            True if valid, False otherwise
        """
        valid_chars = set(["A", "C", "G", "T"])
        if self.include_iupac:
            valid_chars.update(self.iupac_codes.keys())

        return all(c.upper() in valid_chars for c in sequence)

    def _extract_kmers(self, sequence: str) -> List[str]:
        """
        Extract k-mers from sequence with specified stride.

        Args:
            sequence: DNA sequence string

        Returns:
            List of k-mer strings
        """
        if self.k == 1:
            return list(sequence)

        kmers = []
        for i in range(0, len(sequence) - self.k + 1, self.stride):
            kmers.append(sequence[i : i + self.k])

        return kmers

    def encode(self, sequence: str, validate: bool = True) -> torch.Tensor:
        """
        Converts a DNA string into a tensor of integers with BOS and EOS tokens.

        Args:
            sequence: DNA sequence string
            validate: Whether to validate sequence before encoding

        Returns:
            Tensor of token IDs
        """
        if validate and not self.validate_sequence(sequence):
            raise ValueError("Invalid sequence: contains characters not in vocabulary")

        sequence = sequence.upper()
        tokens = [self.bos_token_id]

        # Extract k-mers or characters
        kmers = self._extract_kmers(sequence)

        for kmer in kmers:
            tokens.append(self.vocab.get(kmer, self.unk_token_id))

        tokens.append(self.eos_token_id)
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor, remove_special: bool = True) -> str:
        """
        Converts a tensor of integers back into a DNA string.

        Args:
            token_ids: Tensor or list of token IDs
            remove_special: Whether to remove special tokens from output

        Returns:
            Decoded DNA sequence string
        """
        token_list: List[int] = []
        if isinstance(token_ids, torch.Tensor):
            token_list = token_ids.tolist()
        else:
            token_list = list(token_ids)

        special_tokens = {self.bos_token_id, self.eos_token_id, self.pad_token_id}

        decoded_tokens = []
        for tid in token_list:
            if remove_special and tid in special_tokens:
                continue
            decoded_tokens.append(self.inverse_vocab.get(tid, "[UNK]"))

        # For k-mer tokenization, concatenate differently
        if self.k == 1:
            return "".join(decoded_tokens)
        else:
            # For overlapping k-mers (stride=1), take first char of each k-mer + last k-1 chars
            if self.stride == 1 and len(decoded_tokens) > 0:
                result = "".join(kmer[0] for kmer in decoded_tokens if kmer != "[UNK]")
                if decoded_tokens and decoded_tokens[-1] != "[UNK]":
                    result += decoded_tokens[-1][1:]
                return result
            else:
                # For non-overlapping, just concatenate
                return "".join(decoded_tokens)

    def batch_encode(
        self, sequences: List[str], max_len: Optional[int] = None, validate: bool = True
    ) -> torch.Tensor:
        """
        Encodes a batch of sequences with padding.

        Args:
            sequences: List of DNA sequence strings
            max_len: Maximum sequence length (in tokens). If None, uses longest in batch.
            validate: Whether to validate sequences before encoding

        Returns:
            Padded tensor of shape (batch_size, max_len)
        """
        encoded_seqs = [self.encode(seq, validate=validate) for seq in sequences]

        if max_len is None:
            max_len = max(len(seq) for seq in encoded_seqs)

        # Initialize with PAD tokens
        batch_tensor = torch.full(
            (len(sequences), max_len), self.pad_token_id, dtype=torch.long
        )

        for i, seq in enumerate(encoded_seqs):
            length = min(len(seq), max_len)
            batch_tensor[i, :length] = seq[:length]

        return batch_tensor

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

    def save(self, path: str):
        """Save tokenizer configuration and vocabulary."""
        config = {
            "vocab": self.vocab,
            "k": self.k,
            "stride": self.stride,
            "include_iupac": self.include_iupac,
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load tokenizer from saved configuration."""
        with open(path, "r") as f:
            config = json.load(f)

        return cls(
            vocab=config["vocab"],
            k=config.get("k", 1),
            stride=config.get("stride", 1),
            include_iupac=config.get("include_iupac", False),
        )
