# Algorithm Comparison Results

## Tokenization Algorithms

| Algorithm | Vocab Size | Encode Speed (seq/s) | Accuracy (%) | Compression | Avg Tokens |
|-----------|------------|---------------------|--------------|-------------|------------|
| Character (k=1) | 9 | 5480.9 | 100.0 | 1.00x | 1696.1 |
| K-mer Overlap (k=3, s=1) | 69 | 3209.4 | 100.0 | 1.00x | 1694.1 |
| K-mer Non-overlap (k=3, s=3) | 69 | 9337.9 | 100.0 | 2.99x | 566.7 |
| K-mer Large (k=6, s=1) | 4,101 | 3122.5 | 100.0 | 1.00x | 1691.1 |
| BPE (100 merges) | 109 | 89.0 | 100.0 | 2.91x | 582.0 |

## Embedding Algorithms

| Algorithm | Parameters | Model Size (MB) | Forward Time (ms) | Memory (MB) |
|-----------|------------|-----------------|-------------------|-------------|
| Sinusoidal PE | 2,304 | 0.01 | 0.86 | 0.00 |
| Learnable PE | 264,448 | 1.01 | 1.24 | 0.00 |
| RoPE (Rotary) | 2,304 | 0.01 | 2.69 | 0.00 |
| RoPE Complex | 2,304 | 0.01 | 0.97 | 0.00 |
| ALiBi | 2,304 | 0.01 | 0.49 | 0.00 |

## Recommendations

- **Fastest Tokenizer**: K-mer Non-overlap (k=3, s=3) (9337.9 seq/s)
- **Best Compression**: K-mer Non-overlap (k=3, s=3) (2.99x)
- **Most Accurate**: Character (k=1) (100.0%)
- **Smallest Embedder**: Sinusoidal PE (2,304 params)
- **Fastest Embedder**: ALiBi (0.49ms)
