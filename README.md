# PhyloGen

A generative model for predicting antibiotic-resistant bacterial proteomes. Given a wild-type *E. coli* proteome, PhyloGen generates the mutated proteome you would expect to see under ciprofloxacin resistance, conditioned on phylogenetic context.

## What it does

- Takes a wild-type bacterial proteome as input
- Generates a drug-resistant variant with biologically plausible mutations
- Conditions generation on species, antibiotic, and resistance phenotype
- Uses phylogenetic distance between strains to modulate attention
- Taken E. coli as a case study, with the hope that the framework can be extended to other species and antibiotics in the future.

## Structure

- [`data/`](./data/README.md) — dataset preparation pipelines. Start here to understand how raw BV-BRC genomes were processed into wild-type / resistant proteome pairs
- [`phylogen/`](./phylogen/README.md) — model architecture, tokenizer, dataset loader, and training notebooks (pretrain + finetune)
- [`tests/`](./tests/README.md) — example constrained generation script showing how to run inference

## Quickstart

Install dependencies:
```bash
uv sync
```

To run inference, see `test.ipynb` for an example of how to load a checkpoint and generate a mutated proteome given an unmutated input.
