# Data Module

This module provides utilities for loading and preprocessing DNA sequences from FASTA files.

## Dataset Class

### DNADataset
PyTorch Dataset for loading DNA sequences from FASTA files, including the CARD (Comprehensive Antibiotic Resistance Database).

**Features:**
- Automatic tokenization using any tokenizer (K-mer or BPE)
- Sequence length truncation with BOS/EOS preservation
- Invalid sequence filtering
- FASTA header/ID tracking

## Usage

### Loading CARD Dataset

```python
from data import DNADataset, collate_dna_batch
from tokenizer import DNATokenizer
from torch.utils.data import DataLoader

# Create tokenizer
tokenizer = DNATokenizer(k=3, stride=3)

# Load CARD dataset
dataset = DNADataset(
    fasta_file="path/to/card_database.fasta",
    tokenizer=tokenizer,
    max_len=1024,
    filter_invalid=True
)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=lambda batch: collate_dna_batch(batch, tokenizer.pad_token_id)
)

# Iterate through batches
for batch in dataloader:
    # batch shape: (batch_size, seq_len)
    # Process your batch here
    pass
```

### Quick Sequence Loading

For benchmarking or exploration without creating a full Dataset:

```python
from data import load_fasta_sequences

# Load sequences as strings
sequences = load_fasta_sequences(
    fasta_file="path/to/card_database.fasta",
    max_sequences=1000,  # Limit number of sequences
    filter_invalid=True
)

# Use for benchmarking
for seq in sequences:
    tokens = tokenizer.encode(seq)
    # ...
```

### Accessing Sequence Information

```python
# Get raw sequence
raw_seq = dataset.get_sequence(0)

# Get FASTA header/ID
header = dataset.get_header(0)

print(f"Sequence {header}: {raw_seq[:50]}...")
```

## CARD Database

The CARD (Comprehensive Antibiotic Resistance Database) is a curated resource containing:
- Antibiotic resistance genes
- Associated resistance mechanisms
- Molecular sequences

Download from: https://card.mcmaster.ca/download

## Custom Collate Function

The `collate_dna_batch` function handles:
- Padding sequences to the same length within a batch
- Using the tokenizer's pad token ID
- Returning tensors in batch-first format (batch_size, seq_len)

```python
from functools import partial

# Create a partial function with your pad token ID
collate_fn = partial(collate_dna_batch, pad_token_id=tokenizer.pad_token_id)

dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```
