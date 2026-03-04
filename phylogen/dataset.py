import sys
import pickle
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class ProteomeDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer: 'ProteinTokenizer',
        chunk_size: int = 1024,
        overlap: int = 256,
        phylo_pkl: str = None,
        mode: str = "finetune",
        max_samples: int = None,
        start_idx: int = 0,
        use_mutated_only: bool = False,
        cache_dir: str = None,
        force_recompute: bool = False,
        chunk_cache_pkl: str = None,
    ):
        if phylo_pkl is None:
            phylo_pkl = str(Path(__file__).parent.parent / "data" / "gtdb_data" / "ecoli_phylo_distances.pkl")

        df = pd.read_csv(csv_path, dtype={'genome_id': str})

        if mode == "finetune" and use_mutated_only:
            if 'reversions_applied' in df.columns:
                df = df[df['reversions_applied'] == True].reset_index(drop=True)
                print(f"Filtered to {len(df)} genomes with reversions applied (mutated pairs)")
            else:
                print("Warning: 'reversions_applied' column not found — using all rows")

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

        self.phylo_matrix = pickle.load(open(phylo_pkl, "rb"))
        self.num_reps = self.phylo_matrix.shape[0]

        self.genome_to_phylo_idx = {}
        for idx in range(len(self.df)):
            genome_id = self.df.iloc[idx]['genome_id']
            h = int(hashlib.sha256(genome_id.encode()).hexdigest(), 16)
            self.genome_to_phylo_idx[idx] = h % self.num_reps

        # Caching
        if cache_dir is None:
            cache_dir = str(Path(csv_path).parent)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True, parents=True)

        cache_key_parts = [
            Path(csv_path).stem,
            f"mode_{mode}",
            f"chunk{chunk_size}_overlap{overlap}",
            f"mutated_only_{use_mutated_only}",
            f"start{start_idx}_max{max_samples or 'all'}",
        ]
        cache_filename = "_".join(cache_key_parts) + ".chunks.pkl"
        self.cache_path = cache_dir / cache_filename

        # If an explicit chunk cache pkl is provided, use it directly
        if chunk_cache_pkl is not None:
            chunk_cache_pkl = Path(chunk_cache_pkl)
            print(f"Loading chunk cache from explicit path: {chunk_cache_pkl}")
            with open(chunk_cache_pkl, "rb") as f:
                self.chunk_indices = pickle.load(f)
            print(f"→ Loaded {len(self.chunk_indices):,} cached chunks successfully")
            return

        if self.cache_path.exists() and not force_recompute:
            try:
                print(f"Loading cached useful chunks from: {self.cache_path}")
                with open(self.cache_path, "rb") as f:
                    self.chunk_indices = pickle.load(f)
                print(f"→ Loaded {len(self.chunk_indices):,} cached chunks successfully")
                return
            except Exception as e:
                print(f"Cache load failed — recomputing")

        # Compute chunks (mutation-centered)
        self.chunk_indices = []
        print("Computing useful chunks...")
        MAX_CHUNKS_PER_GENOME = 50

        for g_idx in range(len(self.df)):
            print(f"  Processing genome {g_idx+1}/{len(self.df)} ...")
            row = self.df.iloc[g_idx]

            if mode == "finetune":
                cond = ["[SPECIES_ECOLI]", "[CIPRO]", "[RESISTANT]"]
                bos_id = tokenizer.bos_token_id
                sep_id = tokenizer.vocab["[SEP]"]

                # Pure proteome tokens (with <PROT>/</PROT>, NO BOS, NO cond, NO EOS)
                unmut_prot = tokenizer.encode_fast(row['unmutated_proteome'], add_special_tokens=False)
                mut_prot   = tokenizer.encode_fast(row['mutated_proteome'],   add_special_tokens=False)

                mutation_token_pos = [
                    t for t in range(min(len(unmut_prot), len(mut_prot)))
                    if unmut_prot[t] != mut_prot[t]
                ]
                num_mutations = len(mutation_token_pos)
                print(f"    → Encoded | mutations found: {num_mutations}")

                if num_mutations == 0:
                    continue

                cond_ids = torch.tensor([tokenizer.vocab[c] for c in cond], dtype=torch.long)
                # Prefix up to and including SEP
                prefix = torch.cat([
                    torch.tensor([bos_id], dtype=torch.long),
                    cond_ids,
                    unmut_prot,
                    torch.tensor([sep_id], dtype=torch.long)
                ])
                sep_token_pos = len(prefix) - 1

                token_len = len(prefix) + len(mut_prot)   # full length for chunk bounds

                global_mutation_pos = [sep_token_pos + 1 + t for t in mutation_token_pos]

                print(f"    → Generating chunks around {num_mutations} mutations...")

                genome_added = 0
                for mut_pos in global_mutation_pos:
                    for offset in range(-chunk_size//2, chunk_size//2 + 1, self.stride//2):
                        start = max(0, mut_pos - chunk_size//2 + offset)
                        end = start + chunk_size
                        if end > token_len: continue
                        chunk_muts = [p for p in global_mutation_pos if start <= p < end]
                        if len(chunk_muts) == 0: continue
                        if end <= sep_token_pos + 1: continue  # must overlap continuation

                        self.chunk_indices.append((g_idx, start, sep_token_pos))
                        genome_added += 1
                        if genome_added % 5 == 0 or genome_added == 1:
                            print(f"      Added chunk {genome_added} at start={start} (covers {len(chunk_muts)} muts)")
                        if genome_added >= MAX_CHUNKS_PER_GENOME:
                            break
                    if genome_added >= MAX_CHUNKS_PER_GENOME: break

                print(f"    → Genome {g_idx+1} done | added {genome_added} chunks")

        print(f"Saving chunk cache to: {self.cache_path}")
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.chunk_indices, f)
        print(f"→ Saved {len(self.chunk_indices):,} useful chunks")

        print(f"Dataset ready (finetune mode): {len(self.chunk_indices):,} useful chunks from {len(df)} genomes")

    def __getitem__(self, idx):
        genome_idx, chunk_start, global_sep_pos = self.chunk_indices[idx]
        row = self.df.iloc[genome_idx]

        cond = ["[SPECIES_ECOLI]", "[CIPRO]", "[RESISTANT]"]
        bos_id = self.tokenizer.bos_token_id
        sep_id = self.tokenizer.vocab["[SEP]"]

        cond_ids = torch.tensor([self.tokenizer.vocab[c] for c in cond], dtype=torch.long)

        # Pure proteome (continuation)
        unmut_prot = self.tokenizer.encode_fast(row['unmutated_proteome'], add_special_tokens=False)
        mut_prot   = self.tokenizer.encode_fast(row['mutated_proteome'],   add_special_tokens=False)

        sep_tensor = torch.tensor([sep_id], dtype=torch.long)
        bos_tensor = torch.tensor([bos_id], dtype=torch.long)

        # INPUT:  ... unmutated continuation after SEP
        # LABELS: ... mutated   continuation after SEP
        input_full = torch.cat([bos_tensor, cond_ids, unmut_prot, sep_tensor, unmut_prot])
        label_full = torch.cat([bos_tensor, cond_ids, unmut_prot, sep_tensor, mut_prot])

        chunk_end = chunk_start + self.chunk_size
        input_chunk = input_full[chunk_start:chunk_end].clone()
        labels      = label_full[chunk_start:chunk_end].clone()

        rel_sep = global_sep_pos - chunk_start if chunk_start <= global_sep_pos < chunk_end else -1

        pad_len = self.chunk_size - len(input_chunk)
        if pad_len > 0:
            pad = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
            input_chunk = torch.cat([input_chunk, pad])
            labels      = torch.cat([labels, pad])

        phylo_idx = self.genome_to_phylo_idx[genome_idx]
        phylo_dist = torch.tensor(self.phylo_matrix[phylo_idx], dtype=torch.float)

        return {
            'input_ids': input_chunk,
            'labels': labels,
            'phylo_dist': phylo_dist,
            'genome_id': row['genome_id'],
            'sep_pos': torch.tensor([rel_sep], dtype=torch.long),
            'chunk_start_global': torch.tensor([chunk_start], dtype=torch.long),
        }

    def __len__(self):
        return len(self.chunk_indices)

def collate_fn(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'phylo_dist': torch.stack([b['phylo_dist'] for b in batch]),
        'genome_id': [b['genome_id'] for b in batch],
        'sep_pos': torch.tensor([b['sep_pos'].item() for b in batch]),
    }
