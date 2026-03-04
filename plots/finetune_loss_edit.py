import json
import matplotlib.pyplot as plt

with open('logs/finetune_loss_log.json', 'r') as f:
    loss_data = json.load(f)

with open('logs/edit_acc_log.json', 'r') as f:
    edit_data = json.load(f)

loss_steps = [d["step"] for d in loss_data]
loss_values = [d["loss"] for d in loss_data]
avg_loss_values = [d["avg_loss_this_epoch"] for d in loss_data]

edit_steps = [d["step"] for d in edit_data]
edit_acc_values = [d["edit_acc"] for d in edit_data]

# Plot
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(loss_steps, avg_loss_values, 's-', color='red', linewidth=2, label='Avg loss per epoch')
ax1.set_xlabel('Training Step')
ax1.set_ylabel('Loss', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Edit accuracy on right y-axis
ax2 = ax1.twinx()
ax2.plot(edit_steps, edit_acc_values, '^-', color='green', linewidth=2, label='Edit accuracy')
ax2.set_ylabel('Edit Accuracy', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(0, 1.05)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('Training Progress: Loss & Edit Accuracy (Mask-Only-Edits Loss)')
plt.tight_layout()
plt.show()
