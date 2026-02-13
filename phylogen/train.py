import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from dataset import ProteomeDataset, collate_fn
from model import PhyloGen
from tokenizer import ProteinTokenizer

def main(checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")   
    print(f"Using device: {device}")

    # ────────────────────────────────────────────────
    # CONFIG
    # ────────────────────────────────────────────────

    tokenizer_path = Path(__file__).parent.parent / "tokenizer" / "tokenizer.json"
    tokenizer = ProteinTokenizer.load(str(tokenizer_path))

    csv_path = Path(__file__).parent.parent / "data" / "ecoli_processed_pairs" / "ecoli_pairs_concat.csv"
    phylo_pkl_path = Path(__file__).parent.parent / "data" / "gtdb_data" / "ecoli_phylo_distances.pkl"

    epochs = 1
    batch_size = 4
    chunk_size = 1024
    overlap = 256
    max_samples = 10
    use_mutated_only = False

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    save_every_steps = 1000

    # ────────────────────────────────────────────────

    model = PhyloGen(
        vocab_size=tokenizer.vocab_size,
        tokenizer=tokenizer, # now dynamic pad id
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        max_seq_len=2048,
    ).to(device)

    dataset = ProteomeDataset(
        csv_path=str(csv_path),
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        overlap=overlap,
        phylo_pkl=str(phylo_pkl_path),
        mode="pretrain",
        max_samples=max_samples,
        use_mutated_only=use_mutated_only,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        prefetch_factor=2,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # ────────────────────────────────────────────────
    # LOAD CHECKPOINT IF PROVIDED
    # ────────────────────────────────────────────────
    global_step = 0
    start_epoch = 0
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        global_step = checkpoint.get('step', 0)
        start_epoch = checkpoint.get('epoch', 0)
        print(f"\n✓ Loaded checkpoint from {checkpoint_path}")
        print(f"  Resuming from step {global_step}, epoch {start_epoch}\n")
    
    model.train()

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        total_loss = 0
        steps_this_epoch = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
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

            if global_step % save_every_steps == 0:
                ckpt_path = checkpoint_dir / f"phyologen_step_{global_step}.pt"
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': global_step,
                    'epoch': epoch,
                }, ckpt_path)
                print(f"\nCheckpoint saved: {ckpt_path}")

        avg_loss = total_loss / steps_this_epoch
        print(f"Epoch {epoch+1} finished | Avg loss: {avg_loss:.4f} | Steps: {steps_this_epoch}")

        # Save at end of epoch
        ckpt_path = checkpoint_dir / f"phyologen_epoch_{epoch+1}.pt"
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': global_step,
            'epoch': epoch + 1,
        }, ckpt_path)
        print(f"Epoch checkpoint saved: {ckpt_path}")

    print("Training complete.")

if __name__ == "__main__":
    checkpoint_path = None
    
    mp.set_start_method("fork", force=True)
    main(checkpoint_path=checkpoint_path)
