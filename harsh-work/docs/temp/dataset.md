### Key Points on Datasets, Tokenizer, and Embeddings for PhyloGen
- **CARD Fields**: Focuses on AMR-specific data; all fields cover ontologies, genes, mutations, and sequences. Useful ones include sequence (FASTA for DNA), mutation_type (e.g., SNPs for resistance like gyrA), species, antibiotic, and drug_class—ideal for training on resistance mutations.
- **BV-BRC (PATRIC) Fields**: Broad genomic metadata; all fields span taxonomy, isolates, sequences, and AMR. Useful ones: genome_sequence (FASTA), antimicrobial_resistance (phenotypes like Resistant), patric_id (for features), and Newick trees for phylogenetic distances to scale attention.
- **Tokenizer**: A simple character-level one for DNA (A/C/G/T + specials) is feasible; code provided below uses PyTorch for easy integration.
- **Embedding Model**: Start with a basic nn.Embedding layer; code includes learnable embeddings for tokens, trainable in your transformer.
- **Feasibility for Cloud Guy**: These are lightweight (no heavy ML needed yet); run on Colab, use Biopython/DendroPy for data prep. Teammate can handle finetuning later.

#### Task 1: Dataset Fields Overview
CARD and BV-BRC complement each other: CARD for targeted AMR mutations/genes, BV-BRC for full genomes/trees. Download subsets (~1GB) via their sites. Useful fields align with PhyloGen's needs: input sequences (~1k bp), mutation labels for SFT, tree distances for attention, species/antibiotics for filtering.

| Database | All Fields Summary | Useful Fields for PhyloGen |
|----------|--------------------|----------------------------|
| **CARD** | Ontology (ARO/MOBIO/VIRO): ~100+ terms (accessions, synonyms, relationships); Broadstreet: Gene metadata/sequences; Prevalence: Resistomes/variants. | sequence (DNA FASTA), mutation_type (SNPs/indels), species, antibiotic, drug_class, AMR_gene_family, resistance_mechanism. |
| **BV-BRC** | Genome metadata (~70 attributes: organism, isolate, host, sequence, phenotype, project); Features: Annotations; AMR: Phenotypes; Trees: Newick format. | genome_sequence (FASTA), antimicrobial_resistance (Resistant/Susceptible), patric_id (features), organism_name (taxonomy), Newick_tree (distances). |

#### Task 2: Building the Tokenizer
Here's a custom PyTorch tokenizer class for DNA. It maps A/C/G/T to integers (0-3), adds specials ([BOS]=4, [EOS]=5, [PAD]=6, [UNK]=7, [MUT]=8 for mutations). Save/load via JSON for reproducibility.

```python
import torch
import json
from typing import Dict, List

class DNATokenizer:
    def __init__(self, vocab: Dict[str, int] = None):
        self.vocab = vocab or {'A': 0, 'C': 1, 'G': 2, 'T': 3, '[BOS]': 4, '[EOS]': 5, '[PAD]': 6, '[UNK]': 7, '[MUT]': 8}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, sequence: str) -> torch.Tensor:
        tokens = []
        for char in sequence.upper():
            tokens.append(self.vocab.get(char, self.vocab['[UNK]']))
        tokens = [self.vocab['[BOS]']] + tokens + [self.vocab['[EOS]']]
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        return ''.join(self.inverse_vocab.get(int(t), '[UNK]') for t in tokens if int(t) not in [self.vocab['[BOS]'], self.vocab['[EOS]'], self.vocab['[PAD]']])

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.vocab, f)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            vocab = json.load(f)
        return cls(vocab)

# Usage example
tokenizer = DNATokenizer()
seq = "ATCG"
encoded = tokenizer.encode(seq)  # tensor([4, 0, 3, 1, 2, 5])
decoded = tokenizer.decode(encoded)  # "ATCG"
tokenizer.save('dna_vocab.json')
```

Integrate with DataLoader: Pad sequences to max_len (e.g., 1024) using [PAD].

#### Task 3: Building the Embedding Model
A basic embedder using PyTorch's nn.Embedding. Embed_dim=256 (matches your model). Add sinusoidal positional encodings for sequence order. Trainable in your transformer forward pass.

```python
import torch
import torch.nn as nn
import math

class DNAEmbedder(nn.Module):
    def __init__(self, vocab_size: int = 9, embed_dim: int = 256, max_len: int = 1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = self._create_pos_encoding(max_len, embed_dim)

    def _create_pos_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        embedded = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)  # (batch, seq_len, embed_dim)
        embedded += self.pos_encoding[:, :x.size(1), :]  # Add positional
        return embedded

# Usage example
embedder = DNAEmbedder(vocab_size=9)
x = torch.tensor([[0, 3, 1, 2]])  # Batch of one: A T C G
emb = embedder(x)  # Shape: (1, 4, 256)
```

Next steps: Load CARD/BV-BRC data with Biopython, tokenize batches, pass to embedder in your model init. Test on toy sequences before full training.

---

### Comprehensive Guide to Dataset Fields, Tokenizer, and Embeddings for PhyloGen Implementation

This guide expands on the initial overview, providing exhaustive details drawn from official documentation for CARD and BV-BRC (formerly PATRIC). It includes full field lists, relevance to your transformer-based mutation generator, and step-by-step code with explanations tailored for a cloud-focused developer new to AI/ML. The emphasis is on practicality: minimal dependencies (PyTorch, Biopython), local/Colab execution, and integration with your teammate's ML pipeline. All code is modular, testable, and extensible for phylogenetic attention (e.g., injecting tree distances later).

#### Detailed Dataset Research: Fields in CARD and BV-BRC
CARD and BV-BRC are public, downloadable resources optimized for genomic AMR research. CARD (McMaster University) curates resistance-specific data (~6k models, 414 pathogen resistomes as of 2025), ideal for mutation-focused training. BV-BRC (NIAID-funded) offers broader bacterial/viral genomes (~20k+), annotations, and trees, perfect for phylogenetic context. Both use standard formats (FASTA for sequences, TSV/JSON for metadata, Newick for trees), parseable with Biopython/DendroPy.

Downloads:
- CARD: https://card.mcmaster.ca/download (subsets ~1GB; e.g., Broadstreet for genes/mutations).
- BV-BRC: https://www.bv-brc.org/app/download (API/CLI for bulk; e.g., TSV exports via p3-api).

##### CARD: Complete Field Inventory and Project Relevance
CARD's data emphasizes AMR ontologies and sequences, with no native phylogenetic trees (pair with BV-BRC). Key files: Ontology (semantic relations), Broadstreet (core genes/mutations), Prevalence (variants/resistomes). All fields are relational (e.g., via ARO accessions), enabling filtering for fluoroquinolone resistance (gyrA SNPs).

**Full Fields by File Type**:
- **Ontology Files (OBO/OWL/TSV/JSON; ~4.0.1 version)**:
  - Core: term_accession (e.g., ARO:0000001), term_name (e.g., "beta-lactamase"), synonyms (aliases), definition (description).
  - AMR-Specific: drug_class (e.g., "fluoroquinolone"), amr_gene_family (e.g., "gyrA"), resistance_mechanism (e.g., "target alteration"), confers_resistance_to (links to antibiotics), targeted_by_antibiotic (e.g., ciprofloxacin).
  - Mutations: sequence_variant (SNPs/indels), codon_change (e.g., Ser83Leu), translation (protein effect), reference_sequence (wild-type DNA).
  - Other: model_id (RGI predictions), evidence (experimental/curation), version (update timestamp).
- **Broadstreet Data (JSON/TSV/FASTA)**:
  - Gene/Mutation: gene_id, species (e.g., Escherichia coli), antibiotic (e.g., levofloxacin), mutation_type (SNP/indel/insertion), position (genomic coord), sequence (FASTA DNA/protein), codon, translation.
  - Metadata: prevalence (frequency), drug_class, amr_gene_family, resistance_mechanism, model_type (perfect/loose).
- **Prevalence, Genomes, & Variants (TSV/FASTA/JSON/GZ)**:
  - Resistome: pathogen (species), chromosome_id, plasmid_id, wgs_assembly, allele_sequence (variant FASTA), kmer_signature (for detection).
  - Variants: variant_id, reference_allele, alt_allele, frequency, genomic_island (mobile elements).
  - Stats: num_sequences, coverage, prediction_score (RGI output).

**Useful Fields for PhyloGen** (Prioritized for ~5k sequences):
- Primary: sequence (tokenize DNA for input), mutation_type/position (labels for SFT/DPO on realistic mutations), species/antibiotic (filter for gyrA/fluoroquinolones).
- Secondary: drug_class/resistance_mechanism (auxiliary classification task), prevalence (weight rare mutations).
- Why Useful: Directly feeds autoregressive generation; e.g., input wild-type sequence + tree context → output mutated sequence with gyrA SNP.

Example TSV Row (Broadstreet): `gene_id: gyrA_001, species: E. coli, antibiotic: ciprofloxacin, mutation_type: SNP, position: 83, sequence: ATG... (FASTA), codon: TCT→TTG`.

##### BV-BRC (PATRIC): Complete Field Inventory and Project Relevance
BV-BRC's schema is Solr-based (~70+ metadata attributes per genome), with CLI exports (TSV) via p3-api. Covers full genomes/annotations; AMR from phenotypes/panels; trees for distances. No direct SNP fields, but features/AMR regions proxy mutations.

**Full Fields by Category** (From Genome Table/Metadata; Prefixes like `genome.` in CLI):
- **Organism Info (~15 fields)**: genome_id, genome_name, ncbi_taxon_id, genome_status (Complete/Draft), organism_name (e.g., Escherichia coli), strain, serovar, biovar, pathovar, mlst (multi-locus sequence type), culture_collection, type_strain, antimicrobial_resistance (multi: Resistant/Susceptible/Intermediate), antimicrobial_resistance_evidence (Phenotype/AMR Panel/Comment).
- **Isolate Info (~10 fields)**: isolation_site, isolation_source, isolation_comments, collection_date/year, isolation_country, geographic_location, latitude, longitude, altitude, depth, other_environmental.
- **Host Info (~7 fields)**: host_name, host_sex, host_age, host_health_state, body_sample_site, body_sample_subsite, other_clinical.
- **Sequence Info (~10 fields)**: sequencing_status, sequencing_platform (e.g., Illumina), sequencing_depth, assembly_method, chromosomes, plasmids, contigs, sequences (num), genome_length (bp), gc_content (%), patric_cds (count), refseq_cds.
- **Phenotype Info (~9 fields)**: gram_stain, cell_shape, motility, sporulation, temperature_range, optimal_temperature, salinity, oxygen_requirement, habitat, disease.
- **Project Info (~10 fields)**: sequencing_center, completion_date, publication, sra_accession, bioproject_accession, biosample_accession, assembly_accession, genbank_accessions, refseq_accessions.
- **Features Table (~10 fields)**: patric_id, product (e.g., "DNA gyrase subunit A"), location (start-end), sequence_id (contig), refseq_locus_tag, gene_id, plfam_id (protein family), pgfam_id, feature_type (CDS/rRNA/tRNA), na_length, aa_sequence (protein).
- **AMR-Specific (~5 fields + 97 antibiotics)**: antibiotic_name (e.g., ciprofloxacin), resistant_phenotype (Resistant/Intermediate/Susceptible), molecular_formula, mechanism_of_action, pharmacology_notes. Panels: 27k+ genomes with multi-antibiotic tests.
- **Phylogenetic Trees**: Format: Newick (downloadable); no explicit fields, but metadata includes branch_lengths, node_labels (taxa), support_values (bootstraps). Computed via conserved proteins (e.g., 16S rRNA); up to 100 genomes/tree.

**Useful Fields for PhyloGen** (Prioritized for Tree-Sequence Pairs):
- Primary: genome_sequence (FASTA via contig.sequence), antimicrobial_resistance (labels for classification AUC), patric_id/product (annotate mutations), organism_name/strain (taxonomy for tree grouping).
- Secondary: genome_length/gc_content (normalize inputs), isolation_country/disease (contextual filtering), Newick_tree (parse with DendroPy for dist_matrix: e.g., tree.phylogenetic_distance_matrix()).
- Why Useful: Enables Option A (tree evaluation) immediately; e.g., generate sequences, build trees, compare branch fidelity. For Option B, dist_matrix scales attention.

Example CLI Export Row (Genome-Drug): `genome.genome_id: 83333.1, genome.genome_name: Escherichia coli str. K-12 substr. MG1655, genome_drug.antibiotic: ciprofloxacin, genome_drug.resistant_phenotype: Susceptible`.

| Comparison | CARD Strengths | BV-BRC Strengths | PhyloGen Synergy |
|------------|----------------|------------------|------------------|
| **Sequences/Mutations** | Targeted SNPs (gyrA); FASTA alleles. | Full genomes; feature locations. | CARD for mutations, BV-BRC for context. |
| **AMR Labels** | Drug_class/mechanism. | Phenotypes (97 antibiotics). | Combined for DPO preferences. |
| **Phylogeny** | None. | Newick trees (order-level). | BV-BRC for distances. |
| **Size/Access** | ~1GB subsets. | API/TSV bulk. | 5k samples total. |

Prep Tip: Use Biopython's SeqIO.parse for FASTA; filter via pandas on useful fields (e.g., df[df['antibiotic'] == 'ciprofloxacin']).

#### In-Depth: Building the Tokenizer from Scratch
Tokenizers convert raw DNA to integers for model input. For PhyloGen, character-level (not subword like BPE) suffices—DNA alphabet is small (4 letters). This avoids BPE's complexity for beginners. The class above handles encoding/decoding, padding, and specials ([MUT] for synthetic insertions).

**Why Build Custom?** Existing (e.g., HuggingFace) add overhead; yours is lightweight (~10 lines core). Extend later for k-mers if needed.

**Integration Steps**:
1. Load data: `from Bio import SeqIO; sequences = [str(rec.seq) for rec in SeqIO.parse('card.fasta', 'fasta')]`.
2. Tokenize dataset: Inherit torch.utils.data.Dataset, use `__getitem__` for encode + pad.
3. Batch: Use torch.nn.utils.rnn.pad_sequence with [PAD].
4. Test: Perplexity on held-out sequences post-training.

Edge Cases: Handle N (ambiguous bases) as [UNK]; uppercase inputs.

#### In-Depth: Building the Embedding Model from Scratch
Embeddings map tokens to dense vectors (e.g., 256-dim), capturing semantics (e.g., A near T in purine-pyrimidine). Positional encodings add order awareness for transformers.

**Why Basic nn.Embedding?** Learnable, no pretraining needed (unlike ESM); scales to your 4-layer model. Sinusoidal PE is fixed/non-learnable, efficient.

**Integration Steps**:
1. Init in PhyloGen: `self.embedder = DNAEmbedder(vocab_size=tokenizer.vocab_size)`.
2. Forward: `emb = self.embedder(x) # To transformer layers`.
3. Train: AdamW on cross-entropy; monitor embedding norms to avoid collapse.
4. Ablate: Compare with/without PE (expect 10-15% perplexity drop without).

Advanced: Later, freeze and finetune with LoRA on CARD mutations. For cloud: Deploy via TorchServe on AWS EC2.

**Validation Metrics**: Embedding quality via cosine similarity on similar mutations (e.g., gyrA variants >0.8 sim).

This setup gets you to Review 0 (base architecture) quickly—prototype on 100 sequences in <1 hour. Share code with teammate for DDP tweaks.

### Key Citations
- [CARD Download and Data Files](https://card.mcmaster.ca/download)
- [BV-BRC Genome Metadata Quick Reference](https://www.bv-brc.org/docs/quick_references/organisms_taxon/genome_metadata.html)
- [BV-BRC Data System Documentation](https://www.bv-brc.org/docs/system_documentation/data.html)
- [BV-BRC CLI Tutorial for Field Exports](https://www.bv-brc.org/docs/cli_tutorial/cli_getting_started.html)
- [BV-BRC AMR Metadata](https://www.bv-brc.org/docs/quick_references/organisms_taxon/antimicrobial_resistance.html)
- [BV-BRC Phylogenetic Tree Service](https://www.bv-brc.org/docs/tutorial/phylogenetic_tree/phylogenetic_tree.html)
- [PyTorch Embedding Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [Sinusoidal Positional Encoding Original Paper](https://arxiv.org/abs/1706.03762)
