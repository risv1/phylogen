# Embedding Module

This module contains various DNA embedding algorithms with positional encoding and benchmarking tools.

## Embedding Implementations

### 1. Sinusoidal Embedder (`sinusoidal_embedder.py`)
Traditional transformer-style positional encoding:
- **Sinusoidal PE**: Fixed sine/cosine positional encodings (not learnable)
- **Learnable PE**: Learnable positional embeddings

### 2. ALiBi Embedder (`alibi_embedder.py`)
Attention with Linear Biases - no positional embeddings:
- Adds linear bias to attention scores based on position distance
- Better extrapolation to longer sequences
- Used in modern language models like BLOOM

Reference: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (Press et al., 2022)

### 3. RoPE Embedder (`rope_embedder.py`)
Rotary Position Embedding:
- Rotates embedding dimensions based on position
- Used in LLaMA, GPT-NeoX, and other modern LLMs
- Two implementations: standard and complex number-based

Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)

## Benchmarking

Run the benchmark suite to compare all embedders:

```bash
cd embedding
python benchmark.py --output ../benchmarks/embedding --vocab-size 9 --embed-dim 256 --max-len 1024
```

### Output

The benchmark generates:
- **embedding_comparison.png**: Visual comparison charts
- **embedding_results.md**: Summary table in markdown format
- **embedding_results.json**: Raw results in JSON format

### Metrics

- Parameter count
- Model size (MB)
- Forward pass time (ms)
- Peak memory usage (MB)

## Example Usage

```python
from sinusoidal_embedder import DNAEmbedder
from alibi_embedder import ALiBiEmbedder
from rope_embedder import RoPEEmbedder
import torch

# Sinusoidal embedder
embedder = DNAEmbedder(vocab_size=9, embed_dim=256, pos_type="sinusoidal")
tokens = torch.randint(0, 9, (32, 512))  # batch_size=32, seq_len=512
embedded = embedder(tokens)  # (32, 512, 256)

# ALiBi embedder
embedder = ALiBiEmbedder(vocab_size=9, embed_dim=256, num_heads=8)
embedded = embedder(tokens)

# RoPE embedder
embedder = RoPEEmbedder(vocab_size=9, embed_dim=256)
embedded = embedder(tokens)
```
