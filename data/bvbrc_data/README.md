# BV-BRC Data

Install [BV-BRC-CLI](https://github.com/BV-BRC/BV-BRC-CLI/releases), once inside use `p3-login` for your account.

Once this is done, run
```shell
# 500 E. coli genomes resistant to ciprofloxacin (adjust head for more/less)
p3-echo -t antibiotic ciprofloxacin | p3-get-drug-genomes --resistant --eq "genome_name,Escherichia coli" | head -501 > ecoli_cipro_500_ids.tsv  # Includes header
```
to get the E.Coli samples resistant to ciprofloxacin (500 taken here)

Next,
```shell
# Extract IDs (skip header)
cut -f2 ecoli_cipro_500_ids.tsv | tail -n +2 > genome_ids_only.txt
```
to extract ids

Then, run `download-fasta.sh` to get the FASTA files based on extracted genome ids

After that, to get full genome data, run the following
```shell
p3-get-genome-data --input genome_ids_only.txt --nohead \
  --attr genome_id \
  --attr genome_name \
  --attr taxon_id \
  --attr taxon_lineage_names \
  --attr genome_length \
  --attr antimicrobial_resistance \
  --attr antimicrobial_resistance_evidence > pre_ecoli_metadata.tsv
```

This will give a metadata tsv (will not contain header, removing `--nohead` gets problems in the formatting of the tsv for the first genome record, run the below after)
```shell
HEADER="genome_id\tgenome_name\ttaxon_id\ttaxon_lineage_names\tgenome_length\tantimicrobial_resistance\tantimicrobial_resistance_evidence"

echo -e "$HEADER" > ecoli_metadata_full.tsv
cat pre_ecoli_metadata.tsv >> ecoli_metadata_full.tsv
```

Now after getting info from [CARD](../card_data/README.md), we can see the most common mutations for fluoroquinolone resistance is in gyrA, parC, and parE, so those are the mutations we will be referencing here (there are many more, but these are the mutations we are considering as examples) \
Fetch the information for genome features for each `genome_id`
```shell
p3-get-genome-features --input genome_ids_only.txt --nohead \
  --eq feature_type,CDS \
  --keyword "gyrase subunit A" \
  --attr genome_id \
  --attr patric_id \
  --attr accession \
  --attr start \
  --attr end \
  --attr strand \
  --attr gene \
  --attr product \
  --attr na_sequence \
  --selective \
  > gyrA_features.tsv

p3-get-genome-features --input genome_ids_only.txt --nohead \
  --eq feature_type,CDS \
  --keyword "topoisomerase IV" \
  --attr genome_id \
  --attr patric_id \
  --attr accession \
  --attr start \
  --attr end \
  --attr strand \
  --attr gene \
  --attr product \
  --attr na_sequence \
  --selective \
  > parC_parE_features.tsv
```
