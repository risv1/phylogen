# PhyloGen Algorithm Benchmark Suite

Comprehensive comparison of 5 tokenization and 5 embedding algorithms for DNA sequence modeling.

## 📊 Algorithms Tested

### Tokenization (5 algorithms)

1. **Character-level (k=1)** - Baseline single nucleotide tokenization
2. **K-mer Overlapping (k=3, stride=1)** - Codon-level with overlap
3. **K-mer Non-overlapping (k=3, stride=3)** - Codon-level without overlap
4. **K-mer Large (k=6)** - Larger motif capture
5. **BPE (Byte Pair Encoding)** - Data-driven learned tokenization

### Embedding (5 algorithms)

1. **Sinusoidal PE** - Fixed positional encoding (Transformer original)
2. **Learnable PE** - Trainable position embeddings (BERT-style)
3. **RoPE (Rotary)** - Rotary position embeddings (LLaMA/GPT-NeoX)
4. **RoPE Complex** - Complex number implementation of RoPE
5. **ALiBi** - Attention with Linear Biases (no position embeddings)

## 🚀 Quick Start

### Option 1: Python Runner (Recommended)
```bash
python run_benchmark.py
```

### Option 2: Shell Script (Unix/macOS/Linux)
```bash
./run_benchmark.sh
```

### Option 3: Direct Python
```bash
python src/algorithms/benchmark.py --fasta card-data/nucleotide_fasta_protein_variant_model.fasta
```

## 📋 Benchmark Metrics

### Tokenizer Metrics
- **Vocabulary Size**: Total unique tokens
- **Encoding Speed**: Sequences processed per second
- **Decoding Speed**: Token-to-sequence conversion rate
- **Reconstruction Accuracy**: Exact match after encode/decode roundtrip
- **Compression Ratio**: Average characters per token (higher = better)
- **Average Tokens**: Mean tokens per sequence

### Embedder Metrics
- **Parameter Count**: Total trainable parameters
- **Model Size**: Memory footprint (MB)
- **Forward Pass Time**: Inference latency (ms)
- **Memory Usage**: Peak GPU/CPU memory during forward pass
- **Output Shape**: Embedding tensor dimensions

## 📁 Output Files

After running benchmark, results are saved to `results/benchmarks/`:

```
results/benchmarks/
├── algorithm_comparison.png      # 📊 Visual comparison graphs (9 subplots)
├── COMPARISON_RESULTS.md        # 📄 Markdown tables with metrics
└── benchmark_results.json       # 💾 Raw data in JSON format
```

## 🔧 Advanced Usage

### Custom FASTA File
```bash
python run_benchmark.py --fasta /path/to/sequences.fasta
```

### Adjust Test Size
```bash
python run_benchmark.py --max-seqs 1000  # Test on 1000 sequences
```

### Custom Output Directory
```bash
python run_benchmark.py --output my_results/
```

### Different Embedding Dimension
```bash
python run_benchmark.py --embed-dim 512
```

### All Options
```bash
python run_benchmark.py \
    --fasta card-data/nucleotide_fasta_protein_variant_model.fasta \
    --output results/benchmarks \
    --max-seqs 500 \
    --embed-dim 256 \
    --skip-install  # Skip dependency check
```

## 📦 Dependencies

Install required packages:
```bash
pip install -r requirements_benchmark.txt
```

Or manually:
```bash
pip install torch numpy biopython matplotlib seaborn pandas tqdm
```

## 🔬 Interpreting Results

### Tokenizer Selection Guide

**For Speed-Critical Applications:**
- Choose: Character-level (k=1) or K-mer Non-overlapping
- Fastest encoding/decoding, simplest implementation

**For Biological Accuracy:**
- Choose: K-mer Overlapping (k=3) or BPE
- Captures codon structure and motifs

**For Compression:**
- Choose: BPE or K-mer Large (k=6)
- Fewer tokens per sequence, smaller vocab in some cases

**For Exact Reconstruction:**
- All should achieve 100% accuracy on valid ACGT sequences
- Watch for edge cases with BPE on unseen patterns

### Embedder Selection Guide

**For Smallest Model:**
- Choose: Sinusoidal PE or RoPE
- No learnable position parameters

**For Long Sequences (>1024 bp):**
- Choose: ALiBi or RoPE
- Better extrapolation beyond training length

**For Performance:**
- Choose: Sinusoidal PE or RoPE Complex
- Fastest forward pass, precomputed encodings

**For Adaptability:**
- Choose: Learnable PE
- Position embeddings adapt to data during training

**For Modern LLM Compatibility:**
- Choose: RoPE
- Used in LLaMA, Mistral, GPT-NeoX

## 📊 Example Results Preview

```
TOKENIZER BENCHMARKS
════════════════════════════════════════════════════════════
Algorithm               Vocab    Speed    Accuracy  Compression
────────────────────────────────────────────────────────────
Character (k=1)         9        150/s    100.0%    1.00x
K-mer Overlap (k=3,s=1) 69       120/s    100.0%    0.33x
K-mer Non-overlap       69       140/s    100.0%    0.33x
K-mer Large (k=6,s=1)   4,096    90/s     100.0%    0.17x
BPE (100 merges)        ~109     100/s    99.8%     0.45x

EMBEDDER BENCHMARKS
════════════════════════════════════════════════════════════
Algorithm          Parameters  Size(MB)  Forward(ms)  Memory
────────────────────────────────────────────────────────────
Sinusoidal PE      2,304       0.01      1.2          8.5
Learnable PE       264,448     1.01      1.5          10.2
RoPE (Rotary)      2,304       0.01      1.8          8.7
RoPE Complex       2,304       0.01      1.6          8.8
ALiBi              2,304       0.01      1.1          8.4
```

## 🧪 Validation

The benchmark validates:
- ✅ Encode/decode roundtrip accuracy
- ✅ Batch processing capability
- ✅ Edge cases (short sequences, long sequences)
- ✅ Memory efficiency
- ✅ Inference speed

## 🤝 Contributing

To add new algorithms:

1. Create tokenizer in `src/algorithms/your_tokenizer.py`
2. Implement `encode()`, `decode()`, `batch_encode()` methods
3. Add to `run_all_tokenizers()` in `benchmark.py`
4. Run benchmark and compare!

Same for embedders:
1. Inherit from `nn.Module`
2. Implement `forward(x) -> embeddings`
3. Add to `run_all_embedders()`

## 📖 References

- **BPE**: Sennrich et al. (2016) - Neural Machine Translation of Rare Words
- **Sinusoidal PE**: Vaswani et al. (2017) - Attention Is All You Need
- **RoPE**: Su et al. (2021) - RoFormer: Enhanced Transformer with Rotary Position Embedding
- **ALiBi**: Press et al. (2022) - Train Short, Test Long: Attention with Linear Biases

## 🐛 Troubleshooting

**CARD data not found:**
```bash
# Download CARD dataset first
wget https://card.mcmaster.ca/latest/data -O card-data.tar.bz2
tar -xjf card-data.tar.bz2
```

**Import errors:**
```bash
# Ensure src/ is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**GPU memory issues:**
```bash
# Reduce batch size or sequence count
python run_benchmark.py --max-seqs 100
```

**Matplotlib display issues:**
```bash
# Use Agg backend for headless systems
export MPLBACKEND=Agg
```

## 📈 Future Work

- [ ] Add WordPiece tokenization
- [ ] Add Relative Position Embeddings (T5-style)
- [ ] Add Nucleotide property embeddings (physicochemical)
- [ ] Benchmark on multiple datasets (BV-BRC, RefSeq)
- [ ] Add downstream task evaluation (phylogenetic accuracy)
- [ ] Memory profiling with tracemalloc
- [ ] Multi-GPU benchmarking

---

**Status:** ✅ Production Ready  
**Last Updated:** January 2025  
**Maintainer:** PhyloGen Team