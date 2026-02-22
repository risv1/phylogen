import sys
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import json

sys.path.append(str(Path(__file__).parent.parent))

from dataset import ProteomeDataset, collate_fn
from model import PhyloGen
from tokenizer import ProteinTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

tokenizer_path = Path(__file__).parent.parent / "tokenizer" / "tokenizer.json"
tokenizer = ProteinTokenizer.load(str(tokenizer_path))

csv_path = Path(__file__).parent.parent / "data" / "ecoli_processed_pairs" / "ecoli_pairs_concat.csv"
phylo_pkl_path = Path(__file__).parent.parent / "data" / "gtdb_data" / "ecoli_phylo_distances.pkl"

PRETRAINED_CHECKPOINT = "checkpoints_pretrain/pretrain_epoch_3.pt"

epochs = 1
batch_size = 4
chunk_size = 1024
overlap = 256
lr = 2e-5
use_mutated_only = True
max_samples = None

checkpoint_dir = Path("checkpoints_finetune")
checkpoint_dir.mkdir(exist_ok=True, parents=True)

loss_log_file = Path("finetune_loss_log.json")

save_every_steps = 2000
freeze_first_n_blocks = 3

resume_from = None

model = PhyloGen(
    vocab_size=tokenizer.vocab_size,
    tokenizer=tokenizer,
    embed_dim=256,
    num_heads=8,
    num_layers=6,
    max_seq_len=2048,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

start_epoch = 0
global_step = 0

if loss_log_file.exists():
    with open(loss_log_file, "r") as f:
        loss_log = json.load(f)
else:
    loss_log = []
    with open(loss_log_file, "w") as f:
        json.dump(loss_log, f, indent=4)

if resume_from:
    ckpt = torch.load(resume_from, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt.get('epoch', 1) - 1
    global_step = ckpt.get('step', 0)
    print(f"Resumed at epoch {start_epoch + 1} (1-based), global step {global_step}")
else:
    ckpt = torch.load(PRETRAINED_CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt['model'])
    print("Pretrained base loaded.")

if freeze_first_n_blocks > 0:
    for i, block in enumerate(model.blocks):
        if i < freeze_first_n_blocks:
            for param in block.parameters():
                param.requires_grad = False
    print(f"Froze first {freeze_first_n_blocks} blocks.")

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
    use_mutated_only=use_mutated_only,
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

for epoch in range(start_epoch, epochs):
    epoch_num = epoch + 1
    is_resumed_start = resume_from and (epoch == start_epoch)

    print(f"\n{'Resuming ' if is_resumed_start else ''}Finetune Epoch {epoch_num}/{epochs} "
          f"(global step {global_step})")

    total_loss = 0
    steps_this_epoch = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch_num}/{epochs}{' (resumed)' if is_resumed_start else ''}")
    for batch in pbar:
        global_step += 1
        steps_this_epoch += 1

        input_ids = batch['input_ids'].to(device)
        phylo = batch['phylo_dist'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type):
            out = model(input_ids, phylo, labels=labels)
            loss = out["loss"]

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        loss_log.append({
            "epoch": epoch_num,
            "step": global_step,
            "loss": loss.item(),
            "avg_loss_this_epoch": total_loss / steps_this_epoch if steps_this_epoch > 0 else 0,
        })

        if global_step % save_every_steps == 0:
            ckpt_path = checkpoint_dir / f"finetune_step_{global_step}.pt"
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': global_step,
                'epoch': epoch_num,
            }, ckpt_path)
            print(f"\nCheckpoint saved: {ckpt_path}")

            with open(loss_log_file, "w") as f:
                json.dump(loss_log, f, indent=4)

    avg_loss = total_loss / steps_this_epoch if steps_this_epoch > 0 else 0
    print(f"Epoch {epoch_num} finished | Avg loss: {avg_loss:.4f} | Steps this epoch: {steps_this_epoch}")

    ckpt_path = checkpoint_dir / f"finetune_epoch_{epoch_num}.pt"
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': global_step,
        'epoch': epoch_num,
    }, ckpt_path)
    print(f"Epoch checkpoint saved: {ckpt_path}")

    with open(loss_log_file, "w") as f:
        json.dump(loss_log, f, indent=4)

print("Finetuning complete.")