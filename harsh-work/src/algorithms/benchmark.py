"""
Algorithm Benchmark & Comparison Framework for PhyloGen

Compares 5 tokenization and 5 embedding algorithms on:
- Vocabulary size
- Parameter count
- Encoding/decoding speed
- Memory usage
- Reconstruction accuracy
- Compression ratio

Generates comparative analysis graphs and metrics tables.
"""

import json

# Import tokenizers
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from Bio import SeqIO

sys.path.append(str(Path(__file__).parent.parent))

from algorithms.alibi_embedder import ALiBiEmbedder
from algorithms.bpe_tokenizer import BPETokenizer
from algorithms.model import DNAEmbedder
from algorithms.rope_embedder import RoPEEmbedder, RoPEEmbedderAlternative
from algorithms.tokenizer import DNATokenizer


class AlgorithmBenchmark:
    def __init__(
        self,
        fasta_path: str,
        output_dir: str = "results/benchmarks",
        max_sequences: int = 500,
        embed_dim: int = 256,
        max_len: int = 1024,
    ):
        """
        Initialize benchmark framework.

        Args:
            fasta_path: Path to FASTA file for testing
            output_dir: Directory to save results
            max_sequences: Max number of sequences to test
            embed_dim: Embedding dimension
            max_len: Maximum sequence length
        """
        self.fasta_path = fasta_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_sequences = max_sequences
        self.embed_dim = embed_dim
        self.max_len = max_len

        # Load test sequences
        self.sequences = self._load_sequences()
        print(f"✓ Loaded {len(self.sequences)} sequences for benchmarking")

        # Initialize results storage
        self.tokenizer_results = {}
        self.embedder_results = {}

    def _load_sequences(self) -> List[str]:
        """Load DNA sequences from FASTA file."""
        sequences = []
        with open(self.fasta_path, "r") as f:
            for record in SeqIO.parse(f, "fasta"):
                seq_str = str(record.seq).upper()
                # Filter valid sequences (ACGT only)
                if all(c in "ACGT" for c in seq_str) and len(seq_str) > 10:
                    sequences.append(seq_str)
                    if len(sequences) >= self.max_sequences:
                        break
        return sequences

    def _count_parameters(self, model: torch.nn.Module) -> int:
        """Count trainable parameters in model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _measure_memory(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure peak memory usage of function."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            result = func(*args, **kwargs)
            memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        else:
            # CPU memory tracking is complex, use 0 as placeholder
            result = func(*args, **kwargs)
            memory_mb = 0.0
        return result, memory_mb

    def benchmark_tokenizer(
        self, name: str, tokenizer: Any, requires_training: bool = False
    ) -> Dict[str, Any]:
        """
        Benchmark a single tokenizer.

        Args:
            name: Tokenizer name
            tokenizer: Tokenizer instance
            requires_training: Whether tokenizer needs training (BPE)

        Returns:
            Dictionary of benchmark metrics
        """
        print(f"\n{'=' * 60}")
        print(f"Benchmarking Tokenizer: {name}")
        print(f"{'=' * 60}")

        results: Dict[str, Any] = {"name": name}

        # Train if needed
        if requires_training:
            print("Training tokenizer...")
            start = time.time()
            tokenizer.train(self.sequences[:100], verbose=False)
            train_time = time.time() - start
            results["train_time"] = train_time
            print(f"  Training time: {train_time:.2f}s")

        # Vocabulary size
        results["vocab_size"] = tokenizer.vocab_size
        print(f"  Vocab size: {tokenizer.vocab_size}")

        # Encoding speed
        start = time.time()
        encoded_seqs = []
        for seq in self.sequences[:100]:
            try:
                encoded = tokenizer.encode(seq, validate=False)
                encoded_seqs.append(encoded)
            except Exception as e:
                print(f"  Warning: Encoding failed for sequence: {e}")
                continue
        encode_time = time.time() - start
        results["encode_time"] = encode_time
        results["encode_speed"] = 100 / encode_time  # seqs/sec
        print(
            f"  Encoding time (100 seqs): {encode_time:.3f}s ({100 / encode_time:.1f} seqs/s)"
        )

        # Decoding speed
        if encoded_seqs:
            start = time.time()
            decoded_seqs = []
            for enc in encoded_seqs:
                decoded = tokenizer.decode(enc)
                decoded_seqs.append(decoded)
            decode_time = time.time() - start
            results["decode_time"] = decode_time
            results["decode_speed"] = len(encoded_seqs) / decode_time
            print(
                f"  Decoding time ({len(encoded_seqs)} seqs): {decode_time:.3f}s ({len(encoded_seqs) / decode_time:.1f} seqs/s)"
            )

            # Reconstruction accuracy
            correct = 0
            total = 0
            for orig, decoded in zip(self.sequences[: len(decoded_seqs)], decoded_seqs):
                if orig == decoded:
                    correct += 1
                total += 1
            accuracy = correct / total if total > 0 else 0.0
            results["reconstruction_accuracy"] = accuracy
            print(
                f"  Reconstruction accuracy: {accuracy * 100:.1f}% ({correct}/{total})"
            )

            # Average token length (compression ratio)
            avg_tokens = np.mean([len(enc) for enc in encoded_seqs])
            avg_seq_len = np.mean(
                [len(seq) for seq in self.sequences[: len(encoded_seqs)]]
            )
            compression = avg_seq_len / avg_tokens if avg_tokens > 0 else 0
            results["avg_tokens_per_seq"] = avg_tokens
            results["compression_ratio"] = compression
            print(f"  Avg tokens/seq: {avg_tokens:.1f}")
            print(f"  Compression ratio: {compression:.2f}x")

        return results

    def benchmark_embedder(
        self, name: str, embedder: torch.nn.Module, vocab_size: int
    ) -> Dict[str, Any]:
        """
        Benchmark a single embedder.

        Args:
            name: Embedder name
            embedder: Embedder module
            vocab_size: Vocabulary size (for test input)

        Returns:
            Dictionary of benchmark metrics
        """
        print(f"\n{'=' * 60}")
        print(f"Benchmarking Embedder: {name}")
        print(f"{'=' * 60}")

        results: Dict[str, Any] = {"name": name}

        # Parameter count
        params = self._count_parameters(embedder)
        results["parameters"] = params
        print(f"  Parameters: {params:,}")

        # Model size (MB)
        model_size_mb = params * 4 / 1024**2  # Assume float32
        results["model_size_mb"] = model_size_mb
        print(f"  Model size: {model_size_mb:.2f} MB")

        # Create test batch
        batch_size = 32
        seq_len = 512
        test_input = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward pass speed
        embedder.eval()
        output = None
        with torch.no_grad():
            # Warmup
            _ = embedder(test_input)

            # Benchmark
            start = time.time()
            for _ in range(10):
                output = embedder(test_input)
            forward_time = (time.time() - start) / 10
            results["forward_time"] = forward_time
            print(f"  Forward pass time: {forward_time * 1000:.2f}ms")

            # Output shape
            if output is not None:
                results["output_shape"] = list(output.shape)
                print(f"  Output shape: {tuple(output.shape)}")

        # Memory usage
        with torch.no_grad():
            _, memory = self._measure_memory(embedder, test_input)
            results["memory_mb"] = memory
            print(f"  Memory usage: {memory:.2f} MB")

        return results

    def run_all_tokenizers(self) -> Dict[str, Dict[str, Any]]:
        """Run benchmarks on all tokenizer algorithms."""
        print("\n" + "=" * 80)
        print("TOKENIZER BENCHMARKS")
        print("=" * 80)

        tokenizers = {
            "Character (k=1)": DNATokenizer(k=1, stride=1),
            "K-mer Overlap (k=3, s=1)": DNATokenizer(k=3, stride=1),
            "K-mer Non-overlap (k=3, s=3)": DNATokenizer(k=3, stride=3),
            "K-mer Large (k=6, s=1)": DNATokenizer(k=6, stride=1),
            "BPE (100 merges)": BPETokenizer(num_merges=100),
        }

        requires_training = {
            "Character (k=1)": False,
            "K-mer Overlap (k=3, s=1)": False,
            "K-mer Non-overlap (k=3, s=3)": False,
            "K-mer Large (k=6, s=1)": False,
            "BPE (100 merges)": True,
        }

        for name, tokenizer in tokenizers.items():
            try:
                results = self.benchmark_tokenizer(
                    name, tokenizer, requires_training[name]
                )
                self.tokenizer_results[name] = results
            except Exception as e:
                print(f"ERROR benchmarking {name}: {e}")
                import traceback

                traceback.print_exc()

        return self.tokenizer_results

    def run_all_embedders(self) -> Dict[str, Dict[str, Any]]:
        """Run benchmarks on all embedder algorithms."""
        print("\n" + "=" * 80)
        print("EMBEDDER BENCHMARKS")
        print("=" * 80)

        # Use vocab size from character-level tokenizer
        vocab_size = 9

        embedders = {
            "Sinusoidal PE": DNAEmbedder(
                vocab_size=vocab_size,
                embed_dim=self.embed_dim,
                max_len=self.max_len,
                pos_type="sinusoidal",
            ),
            "Learnable PE": DNAEmbedder(
                vocab_size=vocab_size,
                embed_dim=self.embed_dim,
                max_len=self.max_len,
                pos_type="learnable",
            ),
            "RoPE (Rotary)": RoPEEmbedder(
                vocab_size=vocab_size,
                embed_dim=self.embed_dim,
                max_len=self.max_len,
            ),
            "RoPE Complex": RoPEEmbedderAlternative(
                vocab_size=vocab_size,
                embed_dim=self.embed_dim,
                max_len=self.max_len,
            ),
            "ALiBi": ALiBiEmbedder(
                vocab_size=vocab_size,
                embed_dim=self.embed_dim,
                max_len=self.max_len,
                num_heads=8,
            ),
        }

        for name, embedder in embedders.items():
            try:
                results = self.benchmark_embedder(name, embedder, vocab_size)
                self.embedder_results[name] = results
            except Exception as e:
                print(f"ERROR benchmarking {name}: {e}")
                import traceback

                traceback.print_exc()

        return self.embedder_results

    def generate_comparison_plots(self):
        """Generate comparative visualization plots."""
        print("\n" + "=" * 80)
        print("GENERATING COMPARISON PLOTS")
        print("=" * 80)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (14, 10)

        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))

        # TOKENIZER PLOTS
        tok_names: List[str] = []
        vocab_sizes: List[int] = []
        encode_speeds: List[float] = []
        accuracies: List[float] = []
        compressions: List[float] = []

        if self.tokenizer_results:
            # Extract data
            tok_names = list(self.tokenizer_results.keys())
            vocab_sizes = [
                self.tokenizer_results[n].get("vocab_size", 0) for n in tok_names
            ]
            encode_speeds = [
                self.tokenizer_results[n].get("encode_speed", 0) for n in tok_names
            ]
            accuracies = [
                self.tokenizer_results[n].get("reconstruction_accuracy", 0) * 100
                for n in tok_names
            ]
            compressions = [
                self.tokenizer_results[n].get("compression_ratio", 0) for n in tok_names
            ]

            # Plot 1: Vocabulary Size
            ax1 = plt.subplot(3, 3, 1)
            bars1 = ax1.bar(
                range(len(tok_names)),
                vocab_sizes,
                color=sns.color_palette("husl", len(tok_names)),
            )
            ax1.set_xticks(range(len(tok_names)))
            ax1.set_xticklabels(tok_names, rotation=45, ha="right", fontsize=8)
            ax1.set_ylabel("Vocabulary Size", fontweight="bold")
            ax1.set_title("Tokenizer Vocabulary Size", fontweight="bold", fontsize=12)
            ax1.set_yscale("log")
            for i, v in enumerate(vocab_sizes):
                ax1.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=7)

            # Plot 2: Encoding Speed
            ax2 = plt.subplot(3, 3, 2)
            bars2 = ax2.bar(
                range(len(tok_names)),
                encode_speeds,
                color=sns.color_palette("husl", len(tok_names)),
            )
            ax2.set_xticks(range(len(tok_names)))
            ax2.set_xticklabels(tok_names, rotation=45, ha="right", fontsize=8)
            ax2.set_ylabel("Sequences/Second", fontweight="bold")
            ax2.set_title("Tokenizer Encoding Speed", fontweight="bold", fontsize=12)
            for i, v in enumerate(encode_speeds):
                ax2.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)

            # Plot 3: Reconstruction Accuracy
            ax3 = plt.subplot(3, 3, 3)
            bars3 = ax3.bar(
                range(len(tok_names)),
                accuracies,
                color=sns.color_palette("husl", len(tok_names)),
            )
            ax3.set_xticks(range(len(tok_names)))
            ax3.set_xticklabels(tok_names, rotation=45, ha="right", fontsize=8)
            ax3.set_ylabel("Accuracy (%)", fontweight="bold")
            ax3.set_title("Reconstruction Accuracy", fontweight="bold", fontsize=12)
            ax3.set_ylim(0, 105)
            for i, v in enumerate(accuracies):
                ax3.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=7)

            # Plot 4: Compression Ratio
            ax4 = plt.subplot(3, 3, 4)
            bars4 = ax4.bar(
                range(len(tok_names)),
                compressions,
                color=sns.color_palette("husl", len(tok_names)),
            )
            ax4.set_xticks(range(len(tok_names)))
            ax4.set_xticklabels(tok_names, rotation=45, ha="right", fontsize=8)
            ax4.set_ylabel("Compression Ratio", fontweight="bold")
            ax4.set_title(
                "Tokenizer Compression (higher=better)", fontweight="bold", fontsize=12
            )
            for i, v in enumerate(compressions):
                ax4.text(i, v, f"{v:.2f}x", ha="center", va="bottom", fontsize=7)

        # EMBEDDER PLOTS
        emb_names: List[str] = []
        params: List[float] = []
        forward_times: List[float] = []
        model_sizes: List[float] = []

        if self.embedder_results:
            # Extract data
            emb_names = list(self.embedder_results.keys())
            params = [
                self.embedder_results[n].get("parameters", 0) / 1000 for n in emb_names
            ]  # in thousands
            forward_times = [
                self.embedder_results[n].get("forward_time", 0) * 1000
                for n in emb_names
            ]  # in ms
            model_sizes = [
                self.embedder_results[n].get("model_size_mb", 0) for n in emb_names
            ]

            # Plot 5: Parameter Count
            ax5 = plt.subplot(3, 3, 5)
            bars5 = ax5.bar(
                range(len(emb_names)),
                params,
                color=sns.color_palette("muted", len(emb_names)),
            )
            ax5.set_xticks(range(len(emb_names)))
            ax5.set_xticklabels(emb_names, rotation=45, ha="right", fontsize=8)
            ax5.set_ylabel("Parameters (thousands)", fontweight="bold")
            ax5.set_title("Embedder Parameter Count", fontweight="bold", fontsize=12)
            for i, v in enumerate(params):
                ax5.text(i, v, f"{v:.1f}K", ha="center", va="bottom", fontsize=7)

            # Plot 6: Forward Pass Time
            ax6 = plt.subplot(3, 3, 6)
            bars6 = ax6.bar(
                range(len(emb_names)),
                forward_times,
                color=sns.color_palette("muted", len(emb_names)),
            )
            ax6.set_xticks(range(len(emb_names)))
            ax6.set_xticklabels(emb_names, rotation=45, ha="right", fontsize=8)
            ax6.set_ylabel("Time (ms)", fontweight="bold")
            ax6.set_title("Embedder Forward Pass Time", fontweight="bold", fontsize=12)
            for i, v in enumerate(forward_times):
                ax6.text(i, v, f"{v:.2f}ms", ha="center", va="bottom", fontsize=7)

            # Plot 7: Model Size
            ax7 = plt.subplot(3, 3, 7)
            bars7 = ax7.bar(
                range(len(emb_names)),
                model_sizes,
                color=sns.color_palette("muted", len(emb_names)),
            )
            ax7.set_xticks(range(len(emb_names)))
            ax7.set_xticklabels(emb_names, rotation=45, ha="right", fontsize=8)
            ax7.set_ylabel("Size (MB)", fontweight="bold")
            ax7.set_title("Embedder Model Size", fontweight="bold", fontsize=12)
            for i, v in enumerate(model_sizes):
                ax7.text(i, v, f"{v:.2f}MB", ha="center", va="bottom", fontsize=7)

        # COMBINED ANALYSIS
        # Plot 8: Tokenizer Trade-offs (Speed vs Compression)
        if self.tokenizer_results:
            ax8 = plt.subplot(3, 3, 8)
            scatter_sizes = [vocab_sizes[i] / 100 for i in range(len(tok_names))]
            scatter = ax8.scatter(
                encode_speeds,
                compressions,
                s=scatter_sizes,
                alpha=0.6,
                c=range(len(tok_names)),
                cmap="viridis",
            )
            for i, name in enumerate(tok_names):
                ax8.annotate(
                    name.split()[0],
                    (encode_speeds[i], compressions[i]),
                    fontsize=7,
                    ha="center",
                )
            ax8.set_xlabel("Encoding Speed (seqs/s)", fontweight="bold")
            ax8.set_ylabel("Compression Ratio", fontweight="bold")
            ax8.set_title(
                "Tokenizer: Speed vs Compression", fontweight="bold", fontsize=12
            )
            ax8.grid(True, alpha=0.3)

        # Plot 9: Embedder Trade-offs (Size vs Speed)
        if self.embedder_results:
            ax9 = plt.subplot(3, 3, 9)
            scatter2 = ax9.scatter(
                forward_times,
                params,
                s=200,
                alpha=0.6,
                c=range(len(emb_names)),
                cmap="plasma",
            )
            for i, name in enumerate(emb_names):
                ax9.annotate(
                    name, (forward_times[i], params[i]), fontsize=7, ha="center"
                )
            ax9.set_xlabel("Forward Time (ms)", fontweight="bold")
            ax9.set_ylabel("Parameters (thousands)", fontweight="bold")
            ax9.set_title("Embedder: Speed vs Size", fontweight="bold", fontsize=12)
            ax9.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / "algorithm_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved comparison plot: {plot_path}")
        plt.close()

    def generate_summary_table(self):
        """Generate markdown summary tables."""
        print("\n" + "=" * 80)
        print("GENERATING SUMMARY TABLES")
        print("=" * 80)

        # Tokenizer table
        tok_md = "# Algorithm Comparison Results\n\n"
        tok_md += "## Tokenization Algorithms\n\n"
        tok_md += "| Algorithm | Vocab Size | Encode Speed (seq/s) | Accuracy (%) | Compression | Avg Tokens |\n"
        tok_md += "|-----------|------------|---------------------|--------------|-------------|------------|\n"

        for name, results in self.tokenizer_results.items():
            vocab = results.get("vocab_size", "N/A")
            speed = results.get("encode_speed", 0)
            acc = results.get("reconstruction_accuracy", 0) * 100
            comp = results.get("compression_ratio", 0)
            tokens = results.get("avg_tokens_per_seq", 0)
            tok_md += f"| {name} | {vocab:,} | {speed:.1f} | {acc:.1f} | {comp:.2f}x | {tokens:.1f} |\n"

        # Embedder table
        tok_md += "\n## Embedding Algorithms\n\n"
        tok_md += "| Algorithm | Parameters | Model Size (MB) | Forward Time (ms) | Memory (MB) |\n"
        tok_md += "|-----------|------------|-----------------|-------------------|-------------|\n"

        for name, results in self.embedder_results.items():
            params = results.get("parameters", 0)
            size = results.get("model_size_mb", 0)
            time_ms = results.get("forward_time", 0) * 1000
            memory = results.get("memory_mb", 0)
            tok_md += (
                f"| {name} | {params:,} | {size:.2f} | {time_ms:.2f} | {memory:.2f} |\n"
            )

        # Recommendations
        tok_md += "\n## Recommendations\n\n"

        if self.tokenizer_results:
            # Best speed
            best_speed = max(
                self.tokenizer_results.items(),
                key=lambda x: x[1].get("encode_speed", 0),
            )
            tok_md += f"- **Fastest Tokenizer**: {best_speed[0]} ({best_speed[1].get('encode_speed', 0):.1f} seq/s)\n"

            # Best compression
            best_comp = max(
                self.tokenizer_results.items(),
                key=lambda x: x[1].get("compression_ratio", 0),
            )
            tok_md += f"- **Best Compression**: {best_comp[0]} ({best_comp[1].get('compression_ratio', 0):.2f}x)\n"

            # Most accurate
            best_acc = max(
                self.tokenizer_results.items(),
                key=lambda x: x[1].get("reconstruction_accuracy", 0),
            )
            tok_md += f"- **Most Accurate**: {best_acc[0]} ({best_acc[1].get('reconstruction_accuracy', 0) * 100:.1f}%)\n"

        if self.embedder_results:
            # Smallest
            smallest = min(
                self.embedder_results.items(),
                key=lambda x: x[1].get("parameters", float("inf")),
            )
            tok_md += f"- **Smallest Embedder**: {smallest[0]} ({smallest[1].get('parameters', 0):,} params)\n"

            # Fastest
            fastest = min(
                self.embedder_results.items(),
                key=lambda x: x[1].get("forward_time", float("inf")),
            )
            tok_md += f"- **Fastest Embedder**: {fastest[0]} ({fastest[1].get('forward_time', 0) * 1000:.2f}ms)\n"

        # Save markdown
        md_path = self.output_dir / "COMPARISON_RESULTS.md"
        with open(md_path, "w") as f:
            f.write(tok_md)
        print(f"✓ Saved summary table: {md_path}")

        # Also save JSON
        json_results = {
            "tokenizers": self.tokenizer_results,
            "embedders": self.embedder_results,
        }
        json_path = self.output_dir / "benchmark_results.json"
        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"✓ Saved JSON results: {json_path}")

    def run_full_benchmark(self):
        """Run complete benchmark suite."""
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + " " * 20 + "PhyloGen Algorithm Benchmark Suite" + " " * 24 + "║")
        print("╚" + "=" * 78 + "╝\n")

        # Run all benchmarks
        self.run_all_tokenizers()
        self.run_all_embedders()

        # Generate outputs
        self.generate_comparison_plots()
        self.generate_summary_table()

        print("\n" + "=" * 80)
        print("✓ BENCHMARK COMPLETE")
        print(f"  Results saved to: {self.output_dir}")
        print("=" * 80 + "\n")


def main():
    """Main entry point for benchmark script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark PhyloGen tokenization and embedding algorithms"
    )
    parser.add_argument(
        "--fasta",
        type=str,
        required=True,
        help="Path to FASTA file for testing",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/benchmarks",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=500,
        help="Maximum sequences to test",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=256,
        help="Embedding dimension",
    )

    args = parser.parse_args()

    # Run benchmark
    benchmark = AlgorithmBenchmark(
        fasta_path=args.fasta,
        output_dir=args.output,
        max_sequences=args.max_seqs,
        embed_dim=args.embed_dim,
    )

    benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()
