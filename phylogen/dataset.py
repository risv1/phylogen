"""

ProteomeDataset: 

- Loads rows
- encodes proteomes on the fly
- chunks long tensors (ex: 1.5M becomes many 1024-token windows with overlap)
- includes phylo dists which is our 20x20 matrix tiled/repeated for our genomes
"""

import os
import sys
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).parent.parent))

from tokenizer import ProteinTokenizer

class ProteomeDataset(Dataset):
    def __init__(
            self,
            csv_path,
            tokenizer: ProteinTokenizer,
            chunk_size: int = 1024,
            overlap: int = 256,
            phylo_pkl: str = None,
            mode: str = "finetune",
            max_samples: int = None
    ):
        if phylo_pkl is None:
            phylo_pkl = str(Path(__file__).parent.parent / "data" / "gtdb_data" / "ecoli_phylo_distances.pkl")
        self.df = pd.read_csv(csv_path, dtype={'genome_id': str})
        if max_samples:
            self.df = self.df.head(max_samples)
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.mode = mode
        self.stride = chunk_size - overlap  # Effective step

        # Load phylo matrix (20x20 np.array)
        self.phylo_matrix = pickle.load(open(phylo_pkl, "rb"))
        self.num_reps = self.phylo_matrix.shape[0]  # 20

        # Precompute chunks per genome (for __len__)
        self.chunk_indices = []  # List of (genome_idx, start_pos) for each chunk
        for genome_idx in range(len(self.df)):
            # Estimate length (use unmutated; mutated similar)
            proteome_str = self.df.iloc[genome_idx]['unmutated_proteome']
            token_len = len(self.tokenizer.encode_fast(proteome_str, add_special_tokens=False, conditioning=None))
            num_chunks = max(1, (token_len - chunk_size) // self.stride + 1)
            for chunk_num in range(num_chunks):
                start = chunk_num * self.stride
                self.chunk_indices.append((genome_idx, start))

    def __len__(self):
        return len(self.chunk_indices)
    

    def __getitem__(self, idx):
        genome_idx, start = self.chunk_indices[idx]
        row = self.df.iloc[genome_idx]

        # Conditioning based on mode
        if self.mode == "finetune":
            cond = ["[SPECIES_ECOLI]", "[CIPRO]", "[RESISTANT]"]
            input_str = row['unmutated_proteome']
            target_str = row['mutated_proteome']
        else:  # pretrain
            cond = None  # Or ["[SUSCEPTIBLE]"]
            input_str = row['unmutated_proteome']  # Self-supervised
            target_str = input_str  # Target = input shifted

        # Encode full (fast ver)
        input_ids = self.tokenizer.encode_fast(input_str, add_special_tokens=True, conditioning=cond)
        target_ids = self.tokenizer.encode_fast(target_str, add_special_tokens=True, conditioning=cond)  # Same cond for align

        # Chunk: Slice window
        end = start + self.chunk_size
        input_chunk = input_ids[start:end]
        target_chunk = target_ids[start:end]  # For pretrain/finetune: same

        # Pad if short (last chunk)
        pad_len = self.chunk_size - len(input_chunk)
        if pad_len > 0:
            input_chunk = torch.cat([input_chunk, torch.full((pad_len,), self.tokenizer.pad_token_id)])
            target_chunk = torch.cat([target_chunk, torch.full((pad_len,), self.tokenizer.pad_token_id)])

        # For next-token: target = input[1:] + PAD, but we'll shift in loss
        # Phylo: Assign random rep dist (tile 20x20 → pick row for this genome)
        phylo_idx = random.randint(0, self.num_reps - 1)  # Or cluster-map later
        phylo_dist = torch.tensor(self.phylo_matrix[phylo_idx], dtype=torch.float)  # (20,); model will broadcast

        return {
            'input_ids': input_chunk.long(),
            'labels': target_chunk.long(),  # For loss: ignore_index=pad_id
            'phylo_dist': phylo_dist,
            'genome_id': row['genome_id'],  # For logging
        }
    
def collate_fn(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'phylo_dist': torch.stack([b['phylo_dist'] for b in batch]),
        'genome_id': [b['genome_id'] for b in batch],
    }

if __name__ == "__main__":
    tokenizer_path = Path(__file__).parent.parent / "tokenizer" / "tokenizer.json"
    tokenizer = ProteinTokenizer.load(str(tokenizer_path))
    csv_path = Path(__file__).parent.parent / "data" / "ecoli_processed_pairs" / "ecoli_pairs_concat.csv"
    ds = ProteomeDataset(
        str(csv_path),
        tokenizer,
        chunk_size=1024,
        overlap=256,
        max_samples=20,  # Test small
    )
    print(f"Dataset size: {len(ds)} chunks")

    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))
    print("Sample batch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")  # (4, 1024)
    print(f"  phylo_dist: {batch['phylo_dist'].shape}")  # (4, 20)
