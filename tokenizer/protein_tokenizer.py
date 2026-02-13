from typing import Dict, List, Optional, Tuple
import json
import torch

class ProteinTokenizer:
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        special_tokens: Optional[List[str]] = None,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
    ):
        if vocab is None:
            vocab = self._build_default_vocab(pad_token, unk_token, bos_token, eos_token, special_tokens or [])

        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.pad_token_id = vocab[pad_token]
        self.unk_token_id = vocab[unk_token]
        self.bos_token_id = vocab[bos_token]
        self.eos_token_id = vocab[eos_token]

    def _build_default_vocab(self, pad_token: str, unk_token: str, bos_token: str, eos_token: str, extra_specials: List[str]) -> Dict[str, int]:
        # Standard amino acids + stop
        standard_aas = list("ACDEFGHIKLMNPQRSTVWY*")
        # Common ambiguous / special
        ambiguous = list("XBJZOU")

        base = standard_aas + ambiguous

        # Protein boundary markers
        boundaries = ["<PROT>", "</PROT>"]

        # Core special tokens
        specials = [
            bos_token, eos_token, pad_token, unk_token,
            "[MUT]", "[RESISTANT]", "[SUSCEPTIBLE]",
            "[CIPRO]", "[FLUOROQUINOLONE]", "[AMPC]", "[MEROPENEM]",  # antibiotics
            "[SPECIES_ECOLI]", "[PHYLO_REF]",
        ]

        # Add user-requested extras
        specials += extra_specials

        all_tokens = base + boundaries + specials

        vocab = {token: i for i, token in enumerate(all_tokens)}
        return vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(
        self,
        sequence: str,
        add_special_tokens: bool = True,
        conditioning: Optional[List[str]] = None,
    ) -> torch.LongTensor:
        """
        Encode a proteome string (with <PROT>...</PROT> markers)

        Args:
            sequence: concatenated proteome string
            add_special_tokens: whether to add [BOS]/[EOS]
            conditioning: list of conditioning tokens e.g. ["[SPECIES_ECOLI]", "[CIPRO]", "[RESISTANT]"]

        Returns:
            torch.LongTensor of token ids
        """
        tokens = []

        if add_special_tokens:
            tokens.append(self.bos_token_id)

        if conditioning:
            for cond in conditioning:
                if cond in self.vocab:
                    tokens.append(self.vocab[cond])
                else:
                    tokens.append(self.unk_token_id)

        # Character-level tokenization, but recognize full special tokens like <PROT>
        i = 0
        while i < len(sequence):
            # Check for known multi-char tokens first (greedy longest match)
            matched = False
            for special in ["<PROT>", "</PROT>"] + [t for t in self.vocab if t.startswith("[")]:
                if sequence[i:].startswith(special):
                    tokens.append(self.vocab[special])
                    i += len(special)
                    matched = True
                    break

            if not matched:
                aa = sequence[i].upper()
                tokens.append(self.vocab.get(aa, self.unk_token_id))
                i += 1

        if add_special_tokens:
            tokens.append(self.eos_token_id)

        return torch.tensor(tokens, dtype=torch.long)
    
    def encode_fast(
        self,
        sequence: str,
        add_special_tokens: bool = True,
        conditioning: Optional[List[str]] = None,
    ) -> torch.LongTensor:
        tokens = []

        if add_special_tokens:
            tokens.append(self.bos_token_id)

        if conditioning:
            for cond in conditioning:
                tokens.append(self.vocab.get(cond, self.unk_token_id))

        # Split on protein boundaries to avoid slow per-char checks
        parts = sequence.split("<PROT>")
        for part in parts:
            if not part.strip():
                continue
            subparts = part.split("</PROT>")
            for i, sub in enumerate(subparts):
                if i < len(subparts) - 1:  # before each </PROT>
                    # Add <PROT> only at start of real protein
                    if sub.strip():
                        tokens.append(self.vocab["<PROT>"])
                    # Tokenize AAs fast
                    for aa in sub.upper():
                        if aa in self.vocab:
                            tokens.append(self.vocab[aa])
                        else:
                            tokens.append(self.unk_token_id)
                    tokens.append(self.vocab["</PROT>"])
                else:
                    # Leftover after last </PROT> — rare, but handle
                    for aa in sub.upper():
                        tokens.append(self.vocab.get(aa, self.unk_token_id))

        if add_special_tokens:
            tokens.append(self.eos_token_id)

        return torch.tensor(tokens, dtype=torch.long)

    def decode(
        self,
        token_ids: torch.Tensor | List[int],
        skip_special: bool = False,
        remove_boundaries: bool = False,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        tokens = []
        for tid in token_ids:
            token = self.inverse_vocab.get(tid, self.unk_token)
            if skip_special and token in {self.bos_token, self.eos_token, self.pad_token}:
                continue
            tokens.append(token)

        text = "".join(tokens)

        if remove_boundaries:
            text = text.replace("<PROT>", "").replace("</PROT>", "")

        return text

    def save(self, path: str):
        config = {
            "vocab": self.vocab,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(
            vocab=config["vocab"],
            pad_token=config["pad_token"],
            unk_token=config["unk_token"],
            bos_token=config["bos_token"],
            eos_token=config["eos_token"],
        )


# ────────────────────────────────────────────────
# Quick test
# ────────────────────────────────────────────────

if __name__ == "__main__":
    tokenizer = ProteinTokenizer()

    print("Vocab size:", tokenizer.vocab_size)
    print("Some important ids:")
    print("  [BOS]:", tokenizer.bos_token_id)
    print("  [CIPRO]:", tokenizer.vocab.get("[CIPRO]"))
    print("  <PROT>:", tokenizer.vocab.get("<PROT>"))
    print("  A:", tokenizer.vocab.get("A"))

    # Example proteome snippet
    example = "<PROT>MATKTTTVNG</PROT><PROT>MFVKLLRSVA</PROT>"

    conditioning = ["[SPECIES_ECOLI]", "[CIPRO]", "[RESISTANT]"]

    encoded = tokenizer.encode(example, conditioning=conditioning)
    print("\nEncoded:", encoded.tolist())

    decoded = tokenizer.decode(encoded, skip_special=True)
    print("Decoded:", decoded)

    tokenizer.save("tokenizer.json")
