import sys
import torch
from pathlib import Path
import json
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from tokenizer import ProteinTokenizer
from phylogen.model import PhyloGen
from phylogen.dataset import ProteomeDataset

# === PATHS ===
BASE_DIR = Path(__file__).parent.parent if "__file__" in globals() else Path.cwd().parent
csv_path       = BASE_DIR / "data" / "ecoli_processed_pairs" / "ecoli_pairs_concat.csv"
phylo_pkl_path = BASE_DIR / "data" / "gtdb_data" / "ecoli_phylo_distances.pkl"
tokenizer_path = BASE_DIR / "tokenizer" / "tokenizer.json"
ckpt_path      = BASE_DIR / "checkpoints_finetune" / "finetune_epoch_3.pt"

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = ProteinTokenizer.load(str(tokenizer_path))

# Load model
print("Loading model architecture...")
model = PhyloGen(
    vocab_size=tokenizer.vocab_size,
    tokenizer=tokenizer,
    embed_dim=256,
    num_heads=8,
    num_layers=6,
    max_seq_len=2048,
)

print("Loading finetune checkpoint...")
ckpt = torch.load(ckpt_path, map_location="cpu")
model_state = ckpt.get('model', ckpt)
model.load_state_dict(model_state, strict=False)
model = model.to(device)
model.eval()
print("Model loaded successfully\n")

# Load small dataset subset for testing (fast)
print("Loading small dataset subset for inference testing...")
chunk_cache_pkl = BASE_DIR / "cache" / "ecoli_pairs_concat_mode_finetune_chunk1024_overlap512_mutated_only_True_start0_maxall.chunks.pkl"
dataset = ProteomeDataset(
    csv_path=str(csv_path),
    tokenizer=tokenizer,
    chunk_size=1024,
    overlap=512,
    phylo_pkl=str(phylo_pkl_path),
    mode="finetune",
    max_samples=5,
    use_mutated_only=True,
    force_recompute=False,
    chunk_cache_pkl=str(chunk_cache_pkl) if chunk_cache_pkl.exists() else None,
)
print(f"Dataset ready: {len(dataset)} chunks from {len(dataset.df)} genomes\n")
