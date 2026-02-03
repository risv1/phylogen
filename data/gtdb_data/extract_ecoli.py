# extract_ecoli_subtree_fixed.py
import os
import sys
import dendropy
import pandas as pd
import pickle

def extract_ecoli_subtree(
    tree_path="bac120.tree",
    taxonomy_path="bac120_taxonomy.tsv",
    output_subtree="ecoli_subtree.nwk",
    output_pdm="ecoli_phylo_distances.pkl"
):
    """
    Extract E. coli subtree from GTDB bac120.tree using taxonomy table.
    Handles header-less taxonomy TSV.
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

    # Parse lineage to get species
    def get_species(lineage):
        if pd.isna(lineage):
            return None
        parts = lineage.split(';')
        for part in parts:
            if part.startswith('s__'):
                return part[3:].strip()  # remove 's__' prefix
        return None

    tax_df['species'] = tax_df['lineage'].apply(get_species)

    # Filter E. coli  
    ecoli_df = tax_df[tax_df['species'].str.contains('Escherichia coli', case=False, na=False)]
    ecoli_accessions = set(ecoli_df['genome'])

    print(f"Found {len(ecoli_accessions)} E. coli genomes in taxonomy file.")

    if len(ecoli_accessions) == 0:
        print("No E. coli found. Check taxonomy file or species name matching.")
        sys.exit(1)

    # Load tree
    print("Loading full tree...")
    tree = dendropy.Tree.get(path=tree_path, schema="newick")
    print(f"Tree has {len(tree.taxon_namespace)} leaves.")

    # Match leaves to E. coli accessions
    matched_leaves = []
    tree_accessions = set(taxon.label for taxon in tree.taxon_namespace)
    
    for acc in ecoli_accessions:
        if acc in tree_accessions:
            matched_leaves.append(acc)
        else:
            # Sometimes GTDB uses different prefix — try fuzzy match (optional)
            for tree_acc in tree_accessions:
                if acc.split('_')[-1] in tree_acc:  # e.g., match GCF part
                    matched_leaves.append(tree_acc)
                    break

    if not matched_leaves:
        print("Error: No matching E. coli accessions found in the tree.")
        print("First few tree labels:", [t.label for t in tree.taxon_namespace][:5])
        print("First few taxonomy genomes:", list(ecoli_accessions)[:5])
        sys.exit(1)

    print(f"Matched {len(matched_leaves)} E. coli leaves in the tree.")

    # Prune tree to matched leaves
    tree.retain_taxa_with_labels(matched_leaves)

    # Save subtree
    tree.write(path=output_subtree, schema="newick")
    print(f"Saved E. coli subtree to {output_subtree}")

    # Compute distance matrix
    print("Computing phylogenetic distance matrix...")
    pdm = tree.phylogenetic_distance_matrix()

    # Save
    with open(output_pdm, "wb") as f:
        pickle.dump(pdm, f)
    print(f"Saved distance matrix to {output_pdm}")

if __name__ == "__main__":
    extract_ecoli_subtree()
