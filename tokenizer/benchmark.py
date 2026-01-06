"""
Tokenizer Benchmark & Comparison Framework

Compares different tokenization algorithms on:
- Vocabulary size
- Encoding/decoding speed
- Memory usage
- Reconstruction accuracy
- Compression ratio

Generates comparative analysis graphs and metrics tables.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data import load_fasta_sequences
from tokenizer.bpe_tokenizer import BPETokenizer
from tokenizer.kmer_tokenizer import KmerTokenizer


class TokenizerBenchmark:
    def __init__(
        self,
        fasta_path: str,
        output_dir: str | None = None,
        max_sequences: int = 500,
    ):
        """
        Initialize tokenizer benchmark framework.

        Args:
            fasta_path: Path to FASTA file for testing
            output_dir: Directory to save results (defaults to project_root/benchmarks/tokenizer)
            max_sequences: Max number of sequences to test
        """
        self.fasta_path = fasta_path
        if output_dir is None:
            output_dir = str(Path(__file__).parent.parent / "benchmarks/tokenizer")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_sequences = max_sequences

        # Load test sequences using shared data utility
        self.sequences = load_fasta_sequences(
            fasta_file=self.fasta_path,
            max_sequences=self.max_sequences,
            filter_invalid=True,
        )
        # Filter sequences that are too short
        self.sequences = [seq for seq in self.sequences if len(seq) > 10]
        print(f"✓ Loaded {len(self.sequences)} sequences for benchmarking")

        # Initialize results storage
        self.results = {}

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

    def run_all_tokenizers(self) -> Dict[str, Dict[str, Any]]:
        """Run benchmarks on all tokenizer algorithms."""
        print("\n" + "=" * 80)
        print("TOKENIZER BENCHMARKS")
        print("=" * 80)

        tokenizers = {
            "Character (k=1)": KmerTokenizer(k=1, stride=1),
            "K-mer Overlap (k=3, s=1)": KmerTokenizer(k=3, stride=1),
            "K-mer Non-overlap (k=3, s=3)": KmerTokenizer(k=3, stride=3),
            "K-mer Large (k=6, s=1)": KmerTokenizer(k=6, stride=1),
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
                self.results[name] = results
            except Exception as e:
                print(f"ERROR benchmarking {name}: {e}")
                import traceback

                traceback.print_exc()

        return self.results

    def generate_comparison_plots(self):
        """Generate comparison plots for tokenizer performance."""
        if not self.results:
            print("No results to plot")
            return

        # Set style
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Tokenizer Algorithm Comparison", fontsize=16, fontweight="bold")

        # Extract data
        names = list(self.results.keys())
        vocab_sizes = [self.results[n].get("vocab_size", 0) for n in names]
        encode_speeds = [self.results[n].get("encode_speed", 0) for n in names]
        decode_speeds = [self.results[n].get("decode_speed", 0) for n in names]
        accuracies = [
            self.results[n].get("reconstruction_accuracy", 0) * 100 for n in names
        ]
        compressions = [self.results[n].get("compression_ratio", 0) for n in names]
        avg_tokens = [self.results[n].get("avg_tokens_per_seq", 0) for n in names]

        # 1. Vocabulary Size
        ax = axes[0, 0]
        bars = ax.bar(range(len(names)), vocab_sizes, color="skyblue", edgecolor="navy")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Vocabulary Size")
        ax.set_title("Vocabulary Size by Tokenizer")
        ax.set_yscale("log")
        for i, (bar, val) in enumerate(zip(bars, vocab_sizes)):
            ax.text(
                i, val, f"{val:,}", ha="center", va="bottom", fontsize=9, rotation=0
            )

        # 2. Encoding Speed
        ax = axes[0, 1]
        bars = ax.bar(
            range(len(names)), encode_speeds, color="lightgreen", edgecolor="darkgreen"
        )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Sequences per Second")
        ax.set_title("Encoding Speed")
        for i, (bar, val) in enumerate(zip(bars, encode_speeds)):
            ax.text(i, val, f"{val:.1f}", ha="center", va="bottom", fontsize=9)

        # 3. Decoding Speed
        ax = axes[0, 2]
        bars = ax.bar(
            range(len(names)), decode_speeds, color="lightcoral", edgecolor="darkred"
        )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Sequences per Second")
        ax.set_title("Decoding Speed")
        for i, (bar, val) in enumerate(zip(bars, decode_speeds)):
            ax.text(i, val, f"{val:.1f}", ha="center", va="bottom", fontsize=9)

        # 4. Reconstruction Accuracy
        ax = axes[1, 0]
        bars = ax.bar(range(len(names)), accuracies, color="plum", edgecolor="purple")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Reconstruction Accuracy")
        ax.set_ylim(0, 105)
        for i, (bar, val) in enumerate(zip(bars, accuracies)):
            ax.text(i, val, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

        # 5. Compression Ratio
        ax = axes[1, 1]
        bars = ax.bar(range(len(names)), compressions, color="gold", edgecolor="orange")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Compression Ratio")
        ax.set_title("Compression Ratio (higher = better compression)")
        for i, (bar, val) in enumerate(zip(bars, compressions)):
            ax.text(i, val, f"{val:.2f}x", ha="center", va="bottom", fontsize=9)

        # 6. Average Tokens per Sequence
        ax = axes[1, 2]
        bars = ax.bar(
            range(len(names)), avg_tokens, color="peachpuff", edgecolor="sienna"
        )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Average Tokens")
        ax.set_title("Average Tokens per Sequence")
        for i, (bar, val) in enumerate(zip(bars, avg_tokens)):
            ax.text(i, val, f"{val:.1f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plot_path = self.output_dir / "tokenizer_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved comparison plot: {plot_path}")

    def generate_summary_table(self):
        """Generate summary table and save as markdown and JSON."""
        if not self.results:
            print("No results to summarize")
            return

        # Create markdown table
        md = "# Tokenizer Benchmark Results\n\n"
        md += "| Tokenizer | Vocab Size | Encode (seqs/s) | Decode (seqs/s) | Accuracy (%) | Compression | Avg Tokens |\n"
        md += "|-----------|------------|-----------------|-----------------|--------------|-------------|-----------|\n"

        for name, res in self.results.items():
            vocab = res.get("vocab_size", 0)
            enc_speed = res.get("encode_speed", 0)
            dec_speed = res.get("decode_speed", 0)
            acc = res.get("reconstruction_accuracy", 0) * 100
            comp = res.get("compression_ratio", 0)
            tokens = res.get("avg_tokens_per_seq", 0)

            md += f"| {name} | {vocab:,} | {enc_speed:.1f} | {dec_speed:.1f} | {acc:.1f} | {comp:.2f}x | {tokens:.1f} |\n"

        md += "\n## Summary\n\n"

        if self.results:
            # Fastest encoding
            fastest = max(
                self.results.items(), key=lambda x: x[1].get("encode_speed", 0)
            )
            md += f"- **Fastest Encoding**: {fastest[0]} ({fastest[1].get('encode_speed', 0):.1f} seqs/s)\n"

            # Best compression
            best_comp = max(
                self.results.items(), key=lambda x: x[1].get("compression_ratio", 0)
            )
            md += f"- **Best Compression**: {best_comp[0]} ({best_comp[1].get('compression_ratio', 0):.2f}x)\n"

            # Most accurate
            best_acc = max(
                self.results.items(),
                key=lambda x: x[1].get("reconstruction_accuracy", 0),
            )
            md += f"- **Most Accurate**: {best_acc[0]} ({best_acc[1].get('reconstruction_accuracy', 0) * 100:.1f}%)\n"

        # Save markdown
        md_path = self.output_dir / "tokenizer_results.md"
        with open(md_path, "w") as f:
            f.write(md)
        print(f"✓ Saved summary table: {md_path}")

        # Also save JSON
        json_path = self.output_dir / "tokenizer_results.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Saved JSON results: {json_path}")

    def run_full_benchmark(self):
        """Run complete benchmark suite."""
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + " " * 20 + "Tokenizer Benchmark Suite" + " " * 33 + "║")
        print("╚" + "=" * 78 + "╝\n")

        # Run all benchmarks
        self.run_all_tokenizers()

        # Generate outputs
        self.generate_comparison_plots()
        self.generate_summary_table()

        print("\n" + "=" * 80)
        print("✓ TOKENIZER BENCHMARK COMPLETE")
        print(f"  Results saved to: {self.output_dir}")
        print("=" * 80 + "\n")


def main():
    """Main entry point for tokenizer benchmark script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark DNA tokenization algorithms"
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
        default=None,
        help="Output directory for results (defaults to project_root/benchmarks/tokenizer)",
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=500,
        help="Maximum sequences to test",
    )

    args = parser.parse_args()

    # Run benchmark
    benchmark = TokenizerBenchmark(
        fasta_path=args.fasta,
        output_dir=args.output,
        max_sequences=args.max_seqs,
    )

    benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()
