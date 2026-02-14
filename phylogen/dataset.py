# dataset.py
import os
import sys
import pickle
import random
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).parent.parent))

from tokenizer import ProteinTokenizer


class ProteomeDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer: ProteinTokenizer,
        chunk_size: int = 1024,
        overlap: int = 256,
        phylo_pkl: str = None,
        mode: str = "finetune",
        max_samples: int = None,
        start_idx: int = 0,
        use_mutated_only: bool = False,
    ):
        if phylo_pkl is None:
            phylo_pkl = str(Path(__file__).parent.parent / "data" / "gtdb_data" / "ecoli_phylo_distances.pkl")

        df = pd.read_csv(csv_path, dtype={'genome_id': str})

        # Optional: filter to only genomes with valid reversions (for finetuning)
        if use_mutated_only:
            if 'reversions_applied' in df.columns:
                df = df[df['reversions_applied'] == True].reset_index(drop=True)
                print(f"Filtered to {len(df)} genomes with reversions applied (mutated pairs)")
            else:
                print("Warning: 'reversions_applied' column not found — using all rows")

        # Optional subsample with start_idx for chunked loading
        if max_samples is not None:
            df = df.iloc[start_idx:start_idx + max_samples].reset_index(drop=True)
            print(f"Loading chunk: samples {start_idx} to {start_idx + len(df)}")
        else:
            df = df.iloc[start_idx:].reset_index(drop=True)

        self.df = df
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.mode = mode
        self.stride = chunk_size - overlap

        # Load phylo matrix
        self.phylo_matrix = pickle.load(open(phylo_pkl, "rb"))
        self.num_reps = self.phylo_matrix.shape[0]

        self.genome_to_phylo_idx = {}
        for idx in range(len(self.df)):
            genome_id = self.df.iloc[idx]['genome_id']
            # Deterministic hash → fixed index
            h = int(hashlib.sha256(genome_id.encode()).hexdigest(), 16)
            phylo_idx = h % self.num_reps
            self.genome_to_phylo_idx[idx] = phylo_idx

        # Precompute chunk indices
        self.chunk_indices = []  # (genome_idx, start_pos)
        for genome_idx in range(len(self.df)):
            proteome_str = self.df.iloc[genome_idx]['unmutated_proteome']
            token_len = len(self.tokenizer.encode_fast(
                proteome_str, add_special_tokens=False, conditioning=None
            ))
            num_chunks = max(1, (token_len - chunk_size) // self.stride + 1)
            for chunk_num in range(num_chunks):
                start = chunk_num * self.stride
                self.chunk_indices.append((genome_idx, start))

        print(f"Dataset initialized: {len(self.df)} genomes → {len(self.chunk_indices)} chunks")

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, idx):
        genome_idx, start = self.chunk_indices[idx]
        row = self.df.iloc[genome_idx]

        # Conditioning
        if self.mode == "finetune":
            cond = ["[SPECIES_ECOLI]", "[CIPRO]", "[RESISTANT]"]
            unmut_str = row['unmutated_proteome']
            mut_str   = row['mutated_proteome']

            # Concatenate: unmutated [SEP] mutated
            full_str = unmut_str + "[SEP]" + mut_str

            # Encode once
            full_ids = self.tokenizer.encode_fast(
                full_str,
                add_special_tokens=True,
                conditioning=cond
            )

            # Find separator position (for loss masking in model)
            sep_token_id = self.tokenizer.vocab.get("[SEP]", -1)
            sep_positions = (full_ids == sep_token_id).nonzero(as_tuple=True)[0]
            sep_pos = sep_positions[0].item() if len(sep_positions) > 0 else -1

            input_ids = full_ids
            labels    = full_ids.clone()

        else:  # pretrain
            cond = None
            input_str = row['unmutated_proteome']
            input_ids = self.tokenizer.encode_fast(
                input_str, add_special_tokens=True, conditioning=cond
            )
            labels = input_ids.clone()
            sep_pos = -1  # no masking

        # Chunk
        end = start + self.chunk_size
        input_chunk = input_ids[start:end]
        target_chunk = labels[start:end]

        # Pad if needed
        pad_len = self.chunk_size - len(input_chunk)
        if pad_len > 0:
            pad_tensor = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
            input_chunk = torch.cat([input_chunk, pad_tensor])
            target_chunk = torch.cat([target_chunk, pad_tensor])

        # Fixed phylo
        phylo_idx = self.genome_to_phylo_idx[genome_idx]
        phylo_dist = torch.tensor(self.phylo_matrix[phylo_idx], dtype=torch.float)

        return {
            'input_ids': input_chunk.long(),
            'labels': target_chunk.long(),
            'phylo_dist': phylo_dist,
            'genome_id': row['genome_id'],
            'sep_pos': sep_pos,  # used in model for loss masking
        }


def collate_fn(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'phylo_dist': torch.stack([b['phylo_dist'] for b in batch]),
        'genome_id': [b['genome_id'] for b in batch],
        'sep_pos': torch.tensor([b['sep_pos'] for b in batch]),  # (B,)
    }
