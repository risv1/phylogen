# Tokenizer Benchmark Results

| Tokenizer | Vocab Size | Encode (seqs/s) | Decode (seqs/s) | Accuracy (%) | Compression | Avg Tokens |
|-----------|------------|-----------------|-----------------|--------------|-------------|-----------|
| Character (k=1) | 9 | 5012.5 | 7066.2 | 100.0 | 1.00x | 1696.1 |
| K-mer Overlap (k=3, s=1) | 69 | 2744.7 | 4513.0 | 100.0 | 1.00x | 1694.1 |
| K-mer Non-overlap (k=3, s=3) | 69 | 8222.5 | 21640.2 | 100.0 | 2.99x | 566.7 |
| K-mer Large (k=6, s=1) | 4,101 | 2645.6 | 4192.1 | 100.0 | 1.00x | 1691.1 |
| BPE (100 merges) | 109 | 83.5 | 19175.7 | 100.0 | 2.91x | 582.0 |

## Summary

- **Fastest Encoding**: K-mer Non-overlap (k=3, s=3) (8222.5 seqs/s)
- **Best Compression**: K-mer Non-overlap (k=3, s=3) (2.99x)
- **Most Accurate**: Character (k=1) (100.0%)
