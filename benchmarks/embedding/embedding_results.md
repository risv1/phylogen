# Embedding Benchmark Results

| Embedder | Parameters | Size (MB) | Forward Time (ms) | Memory (MB) |
|----------|------------|-----------|-------------------|-------------|
| Sinusoidal PE | 2,304 | 0.01 | 1.54 | N/A |
| Learnable PE | 264,448 | 1.01 | 1.42 | N/A |
| ALiBi | 2,304 | 0.01 | 0.53 | N/A |
| RoPE | 2,304 | 0.01 | 3.68 | N/A |
| RoPE (Complex) | 2,304 | 0.01 | 1.11 | N/A |

## Summary

- **Smallest Model**: Sinusoidal PE (2,304 params)
- **Fastest Forward Pass**: ALiBi (0.53ms)
