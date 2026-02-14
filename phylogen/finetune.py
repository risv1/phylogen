import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from dataset import ProteomeDataset, collate_fn
from model import PhyloGen
from tokenizer import ProteinTokenizer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

tokenizer_path = Path(__file__).parent.parent / "tokenizer" / "tokenizer.json"
tokenizer = ProteinTokenizer.load(str(tokenizer_path))

csv_path = Path(__file__).parent.parent / "data" / "ecoli_processed_pairs" / "ecoli_pairs_concat.csv"
phylo_pkl_path = Path(__file__).parent.parent / "data" / "gtdb_data" / "ecoli_phylo_distances.pkl"

PRETRAINED_CHECKPOINT = "checkpoints_pretrain/pretrain_epoch_3.pt"  # your latest good checkpoint

epochs = 1
batch_size = 4
chunk_size = 1024
overlap = 256
lr = 2e-5 # much lower than pretrain
use_mutated_only = True # only ~382 samples with reversions
max_samples = None # all qualifying rows

checkpoint_dir = Path("checkpoints_finetune")
checkpoint_dir.mkdir(exist_ok=True)
save_every_steps = 2000

freeze_first_n_blocks = 3

# ────────────────────────────────────────────────

model = PhyloGen(
    vocab_size=tokenizer.vocab_size,
    tokenizer=tokenizer,
    embed_dim=256,
    num_heads=8,
    num_layers=6,
    max_seq_len=2048,
).to(device)

print(f"Loading pretrained checkpoint: {PRETRAINED_CHECKPOINT}")
checkpoint = torch.load(PRETRAINED_CHECKPOINT, map_location=device)
model.load_state_dict(checkpoint['model'])
print("Pretrained weights loaded.")

if freeze_first_n_blocks > 0:
    for i, block in enumerate(model.blocks):
        if i < freeze_first_n_blocks:
            for param in block.parameters():
                param.requires_grad = False
    print(f"Froze first {freeze_first_n_blocks} blocks.")

# Optimizer — only trainable parameters
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr,
    weight_decay=0.01
)

dataset = ProteomeDataset(
    csv_path=str(csv_path),
    tokenizer=tokenizer,
    chunk_size=chunk_size,
    overlap=overlap,
    phylo_pkl=str(phylo_pkl_path),
    mode="finetune",
    max_samples=max_samples,
    use_mutated_only=use_mutated_only
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=True,
)

model.train()
global_step = 0

for epoch in range(epochs):
    print(f"\nFinetune Epoch {epoch+1}/{epochs}")
    total_loss = 0
    steps_this_epoch = 0

    pbar = tqdm(loader, desc=f"Finetune Epoch {epoch+1}")
    for batch in pbar:
        global_step += 1
        steps_this_epoch += 1

        input_ids = batch['input_ids'].to(device)
        phylo = batch['phylo_dist'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type):
            out = model(
                input_ids,
                phylo,
                labels=labels,
                sep_pos=batch['sep_pos'].to(device)  # ← new
            )
            loss = out["loss"]

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if global_step % save_every_steps == 0:
            ckpt_path = checkpoint_dir / f"finetune_step_{global_step}.pt"
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': global_step,
                'epoch': epoch,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    avg_loss = total_loss / steps_this_epoch
    print(f"Epoch {epoch+1} finished | Avg loss: {avg_loss:.4f} | Steps: {steps_this_epoch}")

    ckpt_path = checkpoint_dir / f"finetune_epoch_{epoch+1}.pt"
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': global_step,
        'epoch': epoch + 1,
    }, ckpt_path)
    print(f"Epoch checkpoint saved: {ckpt_path}")

print("Finetuning complete.")
