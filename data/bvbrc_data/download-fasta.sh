#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p fasta_files

# Loop through each genome ID
while read -r genome_id; do
    echo "Fetching contigs for $genome_id..." >&2   # Print to console (stderr), not file
    
    # Download contig FASTA for this genome and save to separate file
    p3-genome-fasta --contig "$genome_id" > "fasta_files/${genome_id}_contigs.fasta"
    
    # Optional: Add a small delay to avoid rate-limiting (BV-BRC can be strict)
    sleep 1
done < genome_ids_only.txt

echo "All downloads complete. FASTA files saved in fasta_files/" >&2
