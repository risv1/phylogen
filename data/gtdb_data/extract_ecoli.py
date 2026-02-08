import os
import sys
import dendropy
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np

def extract_ecoli_subtree(
    tree_path="bac120.tree",
    taxonomy_path="bac120_taxonomy.tsv",
    output_subtree="ecoli_subtree.nwk",
    output_pdm="ecoli_phylo_distances.pkl",
):
    """
    Extract Escherichia subtree from GTDB bac120.tree using taxonomy table.
    Handles header-less taxonomy TSV. Broadens to Escherichia genus to get more representatives.
    Proceeds even if few leaves matched. NO random/approx fallback.
    """
    # Check files exist
    for p in [tree_path, taxonomy_path]:
        if not os.path.isfile(p):
            print(f"Error: File not found: {p}")
            print("Download from: https://data.gtdb.ecogenomic.org/releases/latest/")
            sys.exit(1)

    # Load taxonomy (NO HEADER)
    print("Loading taxonomy...")
    tax_df = pd.read_csv(taxonomy_path, sep='\t', header=None, names=['genome', 'lineage'])

    print(f"Taxonomy table shape: {tax_df.shape}")
    print("First few rows:\n", tax_df.head())

    # Parse lineage to get genus/species
    def get_genus_species(lineage):
        if pd.isna(lineage):
            return None, None
        parts = lineage.split(';')
        genus = species = None
        for part in parts:
            if part.startswith('g__'):
                genus = part[3:].strip()
            elif part.startswith('s__'):
                species = part[3:].strip()
        return genus, species

    tax_df[['genus', 'species']] = tax_df['lineage'].apply(lambda x: pd.Series(get_genus_species(x)))

    # Filter to genus Escherichia (broaden for more reps, includes coli subdivisions)
    ecoli_df = tax_df[tax_df['genus'].str.contains('Escherichia', case=False, na=False)]
    ecoli_accessions = set(ecoli_df['genome'])

    print(f"Found {len(ecoli_accessions)} Escherichia entries in taxonomy file.")

    if len(ecoli_accessions) == 0:
        print("No Escherichia found. Check taxonomy file or genus name matching.")
        sys.exit(1)

    # Load tree
    print("Loading full tree...")
    tree = dendropy.Tree.get(path=tree_path, schema="newick")
    print(f"Tree has {len(tree.taxon_namespace)} leaves.")

    # Improved matching: Strip prefixes, match core accession (e.g., GCF_003697165.2)
    matched_leaves = []
    tree_labels = {taxon.label: taxon for taxon in tree.taxon_namespace}
    
    for acc in tqdm(ecoli_accessions):
        # Exact match
        if acc in tree_labels:
            matched_leaves.append(tree_labels[acc])
            continue
        # Fuzzy: remove RS_/GB_ prefix, match core
        core_acc = acc.split('_', 2)[-1] if '_' in acc else acc
        for tree_label in tree_labels:
            if core_acc in tree_label:
                matched_leaves.append(tree_labels[tree_label])
                break

    # FIXED: leaf is already Taxon → use leaf.label directly
    matched_labels = [leaf.label for leaf in matched_leaves]
    print(f"Matched {len(matched_leaves)} Escherichia leaves in the tree.")
    print("Sample matched labels:", matched_labels[:5] if matched_labels else "None")

    if len(matched_leaves) < 3:
        print("Warning: Very few matched leaves (<3). Proceeding anyway as requested.")
    elif len(matched_leaves) == 0:
        print("Error: No matches at all. Cannot proceed with subtree extraction.")
        sys.exit(1)

    # Prune tree to matched leaves
    subtree = tree.extract_tree_with_taxa(matched_leaves)

    # Save subtree
    subtree.write(path=output_subtree, schema="newick")
    print(f"Saved Escherichia subtree to {output_subtree}")

    # Compute distance matrix (patristic)
    print("Computing phylogenetic distance matrix...")
    pdm = subtree.phylogenetic_distance_matrix()
    
    # To numpy matrix (labels order)
    label_order = [leaf.label for leaf in subtree.leaf_node_iter()]  # Also fixed here
    dist_matrix = np.zeros((len(label_order), len(label_order)))
    for i, label1 in enumerate(label_order):
        for j, label2 in enumerate(label_order):
            dist_matrix[i, j] = pdm.patristic_distance(label1, label2)

    # Save
    with open(output_pdm, "wb") as f:
        pickle.dump(dist_matrix, f)
    print(f"Saved distance matrix to {output_pdm}")
    print(f"Matrix size: {dist_matrix.shape[0]} x {dist_matrix.shape[0]} (based on matched representatives)")

if __name__ == "__main__":
    extract_ecoli_subtree()