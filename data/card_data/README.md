# CARD Data

Install [CARD](https://card.mcmaster.ca/download) data from the website.

## Explanation

First attempt was to get all protein variant models mentioning "E.Coli" or related, in the description, forgetting to tie in relation to fluoroquinolone/ciprofloxacin, which gave us around 14 returned common E.Coli gene mutations, and when adding ciprofloxacin and fluoroquionolone to the filtering from the result, we got 3 common mutations for cipro resistance (Ciprofloxacin is a fluoroquinolone antibiotic, so descriptions match, and confirmed by CARD's ontology)
Here is the query (one of the many weird ones we tried that got what we wanted) with `jq` 

```shell
jq '[to_entries[] | select(try (.value.model_type == "protein variant model") catch false) | select(try (.value.ARO_category | values | map(.category_aro_name | strings) | any(test("fluoroquinolone|ciprofloxacin"; "i"))) catch false) | select(.value.ARO_description | test("E. coli|Escherichia coli"; "i")) | {ARO_accession: .key, ARO_name: (try .value.ARO_name catch ""), gene: (try (.value.ARO_name | split(" ")[0]) catch ""), description: (try .value.ARO_description catch ""), snps: (try (.value.model_param.snp.param_value | values | map(capture("^(?<original>[A-Z])(?<position>[0-9]+)(?<variant_aa>[A-Z])$"))) catch []), protein_sequence: (try (.value.model_sequences.sequence | values | .[0].protein_sequence.sequence // "") catch "")}]' card.json > cipro_ecoli_mutations.json
```

Once we got these, run against `card_processing.ipynb` to flatten the SNPs returned into a CSV we can use for reference.
