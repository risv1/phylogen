# ────────────────────────────────────────────────
# Merge all batch files into one final TSV
# ────────────────────────────────────────────────

set -euo pipefail

GENOME_IDS_FILE="genome_ids_only.txt"
OUTPUT_DIR="cds_features"
BATCH_SIZE=3
FINAL_TSV="all_cds_features.tsv"

HEADER="genome_id\tpatric_id\taccession\tstart\tend\tstrand\tgene\tproduct"

echo -e "\nMerging all batch files into $FINAL_TSV..."

# Write header once
echo -e "$HEADER" > "$FINAL_TSV"

# Append data from all batch files (skip their headers)
find "$OUTPUT_DIR" -name "*.tsv" -type f | sort -V | while read -r f; do
    if [ -s "$f" ]; then
        tail -n +2 "$f" >> "$FINAL_TSV"
    fi
done

if [ -s "$FINAL_TSV" ]; then
    total_rows=$(wc -l < "$FINAL_TSV")
    echo "→ Final merged file created: $FINAL_TSV"
    echo "  Total rows (incl. header): $total_rows"
    echo "  Expected CDS count: roughly $((total_genomes * 4000)) – $((total_genomes * 5500))"
else
    echo "→ Warning: Final merged file is empty or very small." >&2
fi

echo "Done."
echo "Next step: use $FINAL_TSV in your processing script/notebook to slice na_sequence from contigs."
