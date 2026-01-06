"""
Embedding Benchmark & Comparison Framework

Compares different embedding algorithms on:
- Parameter count
- Model size
- Forward pass speed
- Memory usage

Generates comparative analysis graphs and metrics tables.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pe_embedder import PEEmbedder
from alibi_embedder import ALiBiEmbedder
from rope_embedder import RoPEEmbedder, RoPEEmbedderAlternative


class EmbeddingBenchmark:
    def __init__(
        self,
        output_dir: str = "../benchmarks/embedding",
        vocab_size: int = 9,
        embed_dim: int = 256,
        max_len: int = 1024,
    ):
        """
        Initialize embedding benchmark framework.

        Args:
            output_dir: Directory to save results
            vocab_size: Vocabulary size (for test input)
            embed_dim: Embedding dimension
            max_len: Maximum sequence length
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len

        # Initialize results storage
        self.results = {}

        print(f"✓ Initialized embedding benchmark")
        print(f"  Vocab size: {vocab_size}, Embed dim: {embed_dim}, Max len: {max_len}")

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

    def benchmark_embedder(
        self, name: str, embedder: torch.nn.Module
    ) -> Dict[str, Any]:
        """
        Benchmark a single embedder.

        Args:
            name: Embedder name
            embedder: Embedder module

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
        test_input = torch.randint(0, self.vocab_size, (batch_size, seq_len))

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

    def run_all_embedders(self) -> Dict[str, Dict[str, Any]]:
        """Run benchmarks on all embedder algorithms."""
        print("\n" + "=" * 80)
        print("EMBEDDER BENCHMARKS")
        print("=" * 80)

        embedders = {
            "Sinusoidal PE": PEEmbedder(
                vocab_size=self.vocab_size,
                embed_dim=self.embed_dim,
                max_len=self.max_len,
                pos_type="sinusoidal",
            ),
            "Learnable PE": PEEmbedder(
                vocab_size=self.vocab_size,
                embed_dim=self.embed_dim,
                max_len=self.max_len,
                pos_type="learnable",
            ),
            "ALiBi": ALiBiEmbedder(
                vocab_size=self.vocab_size,
                embed_dim=self.embed_dim,
                max_len=self.max_len,
                num_heads=8,
            ),
            "RoPE": RoPEEmbedder(
                vocab_size=self.vocab_size,
                embed_dim=self.embed_dim,
                max_len=self.max_len,
            ),
            "RoPE (Complex)": RoPEEmbedderAlternative(
                vocab_size=self.vocab_size,
                embed_dim=self.embed_dim,
                max_len=self.max_len,
            ),
        }

        for name, embedder in embedders.items():
            try:
                results = self.benchmark_embedder(name, embedder)
                self.results[name] = results
            except Exception as e:
                print(f"ERROR benchmarking {name}: {e}")
                import traceback

                traceback.print_exc()

        return self.results

    def generate_comparison_plots(self):
        """Generate comparison plots for embedder performance."""
        if not self.results:
            print("No results to plot")
            return

        # Set style
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Embedding Algorithm Comparison", fontsize=16, fontweight="bold")

        # Extract data
        names = list(self.results.keys())
        params = [self.results[n].get("parameters", 0) for n in names]
        model_sizes = [self.results[n].get("model_size_mb", 0) for n in names]
        forward_times = [self.results[n].get("forward_time", 0) * 1000 for n in names]  # Convert to ms
        memory_usage = [self.results[n].get("memory_mb", 0) for n in names]

        # 1. Parameter Count
        ax = axes[0, 0]
        bars = ax.bar(range(len(names)), params, color="skyblue", edgecolor="navy")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Parameter Count")
        ax.set_title("Model Parameters")
        for i, (bar, val) in enumerate(zip(bars, params)):
            ax.text(
                i, val, f"{val:,}", ha="center", va="bottom", fontsize=9, rotation=0
            )

        # 2. Model Size
        ax = axes[0, 1]
        bars = ax.bar(range(len(names)), model_sizes, color="lightgreen", edgecolor="darkgreen")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Size (MB)")
        ax.set_title("Model Size")
        for i, (bar, val) in enumerate(zip(bars, model_sizes)):
            ax.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        # 3. Forward Pass Time
        ax = axes[1, 0]
        bars = ax.bar(range(len(names)), forward_times, color="lightcoral", edgecolor="darkred")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Forward Pass Time (lower = better)")
        for i, (bar, val) in enumerate(zip(bars, forward_times)):
            ax.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        # 4. Memory Usage
        ax = axes[1, 1]
        bars = ax.bar(range(len(names)), memory_usage, color="plum", edgecolor="purple")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Memory (MB)")
        ax.set_title("Peak Memory Usage")
        for i, (bar, val) in enumerate(zip(bars, memory_usage)):
            if val > 0:
                ax.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
            else:
                ax.text(i, 0, "N/A", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plot_path = self.output_dir / "embedding_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved comparison plot: {plot_path}")

    def generate_summary_table(self):
        """Generate summary table and save as markdown and JSON."""
        if not self.results:
            print("No results to summarize")
            return

        # Create markdown table
        md = "# Embedding Benchmark Results\n\n"
        md += "| Embedder | Parameters | Size (MB) | Forward Time (ms) | Memory (MB) |\n"
        md += "|----------|------------|-----------|-------------------|-------------|\n"

        for name, res in self.results.items():
            params = res.get("parameters", 0)
            size = res.get("model_size_mb", 0)
            fwd_time = res.get("forward_time", 0) * 1000
            memory = res.get("memory_mb", 0)

            memory_str = f"{memory:.2f}" if memory > 0 else "N/A"
            md += f"| {name} | {params:,} | {size:.2f} | {fwd_time:.2f} | {memory_str} |\n"

        md += "\n## Summary\n\n"

        if self.results:
            # Smallest
            smallest = min(
                self.results.items(), key=lambda x: x[1].get("parameters", float("inf"))
            )
            md += f"- **Smallest Model**: {smallest[0]} ({smallest[1].get('parameters', 0):,} params)\n"

            # Fastest
            fastest = min(
                self.results.items(),
                key=lambda x: x[1].get("forward_time", float("inf")),
            )
            md += f"- **Fastest Forward Pass**: {fastest[0]} ({fastest[1].get('forward_time', 0) * 1000:.2f}ms)\n"

        # Save markdown
        md_path = self.output_dir / "embedding_results.md"
        with open(md_path, "w") as f:
            f.write(md)
        print(f"✓ Saved summary table: {md_path}")

        # Also save JSON
        json_path = self.output_dir / "embedding_results.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Saved JSON results: {json_path}")

    def run_full_benchmark(self):
        """Run complete benchmark suite."""
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + " " * 20 + "Embedding Benchmark Suite" + " " * 33 + "║")
        print("╚" + "=" * 78 + "╝\n")

        # Run all benchmarks
        self.run_all_embedders()

        # Generate outputs
        self.generate_comparison_plots()
        self.generate_summary_table()

        print("\n" + "=" * 80)
        print("✓ EMBEDDING BENCHMARK COMPLETE")
        print(f"  Results saved to: {self.output_dir}")
        print("=" * 80 + "\n")


def main():
    """Main entry point for embedding benchmark script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark DNA embedding algorithms"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../benchmarks/embedding",
        help="Output directory for results",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=9,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=256,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1024,
        help="Maximum sequence length",
    )

    args = parser.parse_args()

    # Run benchmark
    benchmark = EmbeddingBenchmark(
        output_dir=args.output,
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        max_len=args.max_len,
    )

    benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()
