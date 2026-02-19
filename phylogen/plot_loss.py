import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_loss(log_file, title="Training Loss"):
    with open(log_file, "r") as f:
        loss_log = json.load(f)

    steps = [entry["step"] for entry in loss_log]
    losses = [entry["loss"] for entry in loss_log]
    avg_losses = [entry["avg_loss"] for entry in loss_log]

    plt.figure(figsize=(12, 6))
    plt.plot(steps, losses, label="Per-Step Loss", alpha=0.5)
    plt.plot(steps, avg_losses, label="Running Avg Loss", linewidth=2)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{log_file.stem}_plot.png")
    plt.show()

plot_loss("pretrain_loss_log.json", "Pretrain Loss Curve")
plot_loss("finetune_loss_log.json", "Finetune Loss Curve")
