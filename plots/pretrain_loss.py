import json
import matplotlib.pyplot as plt

with open("logs/pretrain_loss_log.json", "r") as f:
    data = json.load(f)

steps = [entry["step"] for entry in data]
losses = [entry["loss"] for entry in data]
avg_losses = [entry["avg_loss_this_epoch"] for entry in data]
epochs = [entry["epoch"] for entry in data]

plt.figure(figsize=(10, 6))
plt.plot(steps, losses, 'o-', label='Batch loss', color='blue', alpha=0.6)
plt.plot(steps, avg_losses, 's-', label='Avg loss per epoch', color='red', linewidth=2)
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Pretraining Loss Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.show()
