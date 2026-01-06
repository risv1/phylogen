"""
Data utilities module for PhyloGen

Provides dataset classes and utility functions for loading DNA sequences
from FASTA files, including the CARD database.
"""

from .dataset import DNADataset, collate_dna_batch, load_fasta_sequences

__all__ = ["DNADataset", "collate_dna_batch", "load_fasta_sequences"]
