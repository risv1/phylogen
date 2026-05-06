# Data

Raw data lives in `bvbrc_data/`, `card_data/`, and `gtdb_data/`. Processed outputs go to `ecoli_processed_pairs/`.

## Notebooks

### `protein_processing.ipynb`
Takes 500 E. coli ciprofloxacin-resistant genomes from BV-BRC. For each genome, reverts known resistance mutations in gyrA/parC/parE back to wild-type at the DNA level using CARD SNP annotations, then extracts and translates all CDS to produce paired mutated/unmutated protein FASTAs. Outputs `ecoli_pairs.csv` with paths to per-genome `.faa` files (469 valid pairs after filtering genomes with empty FASTAs).

### `proteomes_processing.ipynb`
Reads the per-genome `.faa` files from `protein_processing`, concatenates all proteins into a single string per genome using `<PROT>...</PROT>` delimiters, and stores the result as inline columns `mutated_proteome` / `unmutated_proteome` in `ecoli_pairs_concat.csv`. This is the file consumed by the training datasets.
