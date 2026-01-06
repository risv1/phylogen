"""
BPE (Byte Pair Encoding) Tokenizer for DNA Sequences

Learns common DNA motifs from data by iteratively merging
the most frequent character pairs.
"""

import json
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import torch


class BPETokenizer:
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        num_merges: int = 100,
    ):
        """
        BPE Tokenizer for DNA sequences.

        Args:
            vocab: Pre-built vocabulary mapping tokens to IDs
            merges: List of merge operations (pair1, pair2) -> merged
            num_merges: Number of merge operations to learn (if training)
        """
        self.num_merges = num_merges
        self.merges = merges if merges is not None else []

        if vocab is None:
            # Initialize with base vocabulary
            vocab = self._init_base_vocab()

        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = self.vocab["[PAD]"]
        self.bos_token_id = self.vocab["[BOS]"]
        self.eos_token_id = self.vocab["[EOS]"]
        self.unk_token_id = self.vocab["[UNK]"]

    def _init_base_vocab(self) -> Dict[str, int]:
        """Initialize base vocabulary with nucleotides and special tokens."""
        vocab = {
            "A": 0,
            "C": 1,
            "G": 2,
            "T": 3,
            "[BOS]": 4,
            "[EOS]": 5,
            "[PAD]": 6,
            "[UNK]": 7,
            "[MUT]": 8,
        }
        return vocab

    def _get_pairs(self, word: List[str]) -> Counter:
        """Get all adjacent pairs in a word."""
        pairs = Counter()
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += 1
        return pairs

    def _merge_pair(self, word: List[str], pair: Tuple[str, str]) -> List[str]:
        """Merge all occurrences of a pair in word."""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(word[i] + word[i + 1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return new_word

    def train(self, sequences: List[str], verbose: bool = True):
        """
        Learn BPE merges from sequences.

        Args:
            sequences: List of DNA sequences to learn from
            verbose: Print progress
        """
        # Convert sequences to character lists
        words = [[c for c in seq.upper()] for seq in sequences]

        # Count word frequencies (treat each sequence equally)
        word_freqs = Counter()
        for word in words:
            word_freqs[tuple(word)] += 1

        # Learn merges
        for merge_idx in range(self.num_merges):
            # Count all pairs across all words
            pair_counts = Counter()
            for word, freq in word_freqs.items():
                pairs = self._get_pairs(list(word))
                for pair, count in pairs.items():
                    pair_counts[pair] += count * freq

            if not pair_counts:
                break

            # Get most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]
            self.merges.append(best_pair)

            # Add merged token to vocabulary
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)
                self.inverse_vocab[self.vocab[merged_token]] = merged_token

            # Update word frequencies with merged pairs
            new_word_freqs = Counter()
            for word, freq in word_freqs.items():
                new_word = self._merge_pair(list(word), best_pair)
                new_word_freqs[tuple(new_word)] += freq
            word_freqs = new_word_freqs

            if verbose and (merge_idx + 1) % 10 == 0:
                print(
                    f"Learned {merge_idx + 1}/{self.num_merges} merges, vocab size: {len(self.vocab)}"
                )

        if verbose:
            print(
                f"✓ BPE training complete: {len(self.merges)} merges, vocab size: {len(self.vocab)}"
            )

    def _tokenize(self, sequence: str) -> List[str]:
        """Apply BPE merges to tokenize a sequence."""
        word = list(sequence.upper())

        # Apply all learned merges
        for pair in self.merges:
            word = self._merge_pair(word, pair)

        return word

    def encode(self, sequence: str, validate: bool = True) -> torch.Tensor:
        """
        Encode DNA sequence to token IDs.

        Args:
            sequence: DNA sequence string
            validate: Whether to validate sequence

        Returns:
            Tensor of token IDs
        """
        if validate:
            valid_chars = set(["A", "C", "G", "T"])
            if not all(c.upper() in valid_chars for c in sequence):
                raise ValueError("Invalid sequence: contains non-ACGT characters")

        tokens = [self.bos_token_id]

        # Tokenize using BPE
        bpe_tokens = self._tokenize(sequence)

        for token in bpe_tokens:
            tokens.append(self.vocab.get(token, self.unk_token_id))

        tokens.append(self.eos_token_id)
        return torch.tensor(tokens, dtype=torch.long)

    def decode(
        self, token_ids: Union[torch.Tensor, List[int]], remove_special: bool = True
    ) -> str:
        """
        Decode token IDs back to DNA sequence.

        Args:
            token_ids: Tensor or list of token IDs
            remove_special: Whether to remove special tokens

        Returns:
            Decoded DNA sequence
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        special_tokens = {self.bos_token_id, self.eos_token_id, self.pad_token_id}

        decoded_tokens = []
        for tid in token_ids:
            if remove_special and tid in special_tokens:
                continue
            decoded_tokens.append(self.inverse_vocab.get(tid, "[UNK]"))

        return "".join(decoded_tokens)

    def batch_encode(
        self, sequences: List[str], max_len: Optional[int] = None, validate: bool = True
    ) -> torch.Tensor:
        """
        Encode batch of sequences with padding.

        Args:
            sequences: List of DNA sequences
            max_len: Maximum sequence length in tokens
            validate: Whether to validate sequences

        Returns:
            Padded tensor of shape (batch_size, max_len)
        """
        encoded_seqs = [self.encode(seq, validate=validate) for seq in sequences]

        if max_len is None:
            max_len = max(len(seq) for seq in encoded_seqs)

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
        """Save tokenizer configuration."""
        config = {
            "vocab": self.vocab,
            "merges": self.merges,
            "num_merges": self.num_merges,
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load tokenizer from saved configuration."""
        with open(path, "r") as f:
            config = json.load(f)

        # Convert merges back to tuples
        merges = [tuple(pair) for pair in config.get("merges", [])]

        return cls(
            vocab=config["vocab"],
            merges=merges,
            num_merges=config.get("num_merges", 100),
        )
