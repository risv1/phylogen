# Tokenizer Module

This module contains various DNA tokenization algorithms and benchmarking tools.

## Tokenizer Implementations

### 1. K-mer Tokenizer (`kmer_tokenizer.py`)
Character-level and k-mer based tokenization with configurable parameters:
- **k**: K-mer size (1 for character-level, 3 for codons, etc.)
- **stride**: Stride for k-mer extraction (1 for overlapping, k for non-overlapping)
- **include_iupac**: Include IUPAC ambiguous nucleotide codes

### 2. BPE Tokenizer (`bpe_tokenizer.py`)
Byte Pair Encoding tokenizer that learns common DNA motifs from data.
- Learns merge operations iteratively
- Configurable number of merges
- Better compression for repetitive sequences

## Benchmarking

Run the benchmark suite to compare all tokenizers:

```bash
cd tokenizer
python benchmark.py --fasta <path-to-fasta-file> --output ../benchmarks/tokenizer --max-seqs 500
```

### Output

The benchmark generates:
- **tokenizer_comparison.png**: Visual comparison charts
- **tokenizer_results.md**: Summary table in markdown format
- **tokenizer_results.json**: Raw results in JSON format

### Metrics

- Vocabulary size
- Encoding/decoding speed (sequences per second)
- Reconstruction accuracy
- Compression ratio
- Average tokens per sequence

## Example Usage

```python
from kmer_tokenizer import DNATokenizer
from bpe_tokenizer import BPETokenizer

# K-mer tokenizer (3-mer, non-overlapping)
tokenizer = DNATokenizer(k=3, stride=3)
encoded = tokenizer.encode("ACGTACGT")
decoded = tokenizer.decode(encoded)

# BPE tokenizer (requires training)
bpe = BPETokenizer(num_merges=100)
bpe.train(sequences, verbose=True)
encoded = bpe.encode("ACGTACGT")
decoded = bpe.decode(encoded)
```
