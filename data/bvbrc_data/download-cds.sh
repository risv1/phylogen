#!/bin/bash

# Script: Bulk fetch ALL CDS features in small batches (3 genomes at a time)
#         Saves per-batch TSVs in cds_features/
#         Names files: 1_3.tsv, 4_6.tsv, etc. (1-based line numbers)
#         Merges everything into all_cds_features.tsv at the end
#         Does NOT fetch na_sequence (extract later from contigs)

set -euo pipefail

GENOME_IDS_FILE="genome_ids_only.txt"
OUTPUT_DIR="cds_features"
BATCH_SIZE=3
FINAL_TSV="all_cds_features.tsv"

if [ ! -s "$GENOME_IDS_FILE" ]; then
    echo "Error: $GENOME_IDS_FILE is missing or empty." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

total_genomes=$(wc -l < "$GENOME_IDS_FILE")
echo "Found $total_genomes genomes in $GENOME_IDS_FILE"
echo "Fetching CDS features in batches of $BATCH_SIZE..."

# ────────────────────────────────────────────────
# Prepare header (used for every batch + final file)
# ────────────────────────────────────────────────
HEADER="genome_id\tpatric_id\taccession\tstart\tend\tstrand\tgene\tproduct"

# ────────────────────────────────────────────────
# Process in batches
# ────────────────────────────────────────────────
batch_num=1
start_line=1

while [ "$start_line" -le "$total_genomes" ]; do
    end_line=$((start_line + BATCH_SIZE - 1))
    if [ "$end_line" -gt "$total_genomes" ]; then
        end_line="$total_genomes"
    fi
        
    batch_file="${OUTPUT_DIR}/${start_line}_${end_line}.tsv"
    tmp_file="${batch_file}.tmp"

    echo "Batch $batch_num: lines $start_line - $end_line → $batch_file"

    # Extract batch of genome IDs
    sed -n "${start_line},${end_line}p" "$GENOME_IDS_FILE" > batch.tmp.ids

    # Fetch features for this batch (no na_sequence)
    p3-get-genome-features --input batch.tmp.ids --nohead \
        --eq feature_type,CDS \
        --attr genome_id \
        --attr patric_id \
        --attr accession \
        --attr start \
        --attr end \
        --attr strand \
        --attr gene \
        --attr product \
        --selective > "$tmp_file" 2>/dev/null || {
            echo "Warning: Fetch failed for batch $start_line–$end_line" >&2
            rm -f "$tmp_file" batch.tmp.ids
            continue
        }

    # Write batch file with header only if we got data
    if [ -s "$tmp_file" ]; then
        {
            echo -e "$HEADER"
            cat "$tmp_file"
        } > "$batch_file"
        echo "  → Saved: $batch_file ($(wc -l < "$batch_file") lines incl. header)"
    else
        echo "  → Warning: No CDS features returned for batch $start_line–$end_line"
        touch "$batch_file"  # create empty file so merge doesn't complain
    fi

    rm -f "$tmp_file" batch.tmp.ids

    ((batch_num++))
    start_line=$((end_line + 1))
done