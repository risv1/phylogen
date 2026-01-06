# Embedding Benchmark Results

| Embedder | Parameters | Size (MB) | Forward Time (ms) | Memory (MB) |
|----------|------------|-----------|-------------------|-------------|
| Sinusoidal PE | 2,304 | 0.01 | 1.01 | N/A |
| Learnable PE | 264,448 | 1.01 | 1.72 | 0.02 |
| ALiBi | 2,304 | 0.01 | 0.46 | N/A |
| RoPE | 2,304 | 0.01 | 2.97 | N/A |
| RoPE (Complex) | 2,304 | 0.01 | 1.32 | N/A |

## Summary

- **Smallest Model**: Sinusoidal PE (2,304 params)
- **Fastest Forward Pass**: ALiBi (0.46ms)
