from tbparse import SummaryReader
import matplotlib.pyplot as plt
import numpy as np

log_dir = "logs/bitcoin-alpha-binary-mask_032/version_0"
reader = SummaryReader(log_dir)
df = reader.scalars

# Extract metrics
train_auc = df[df["tag"] == "train_auc_epoch"]
val_auc = df[df["tag"] == "val_auc_epoch"]
train_loss = df[df["tag"] == "train_loss"]
val_loss = df[df["tag"] == "val_loss"]

# Plot AUC over epochs
plt.figure(figsize=(8, 5))
plt.plot(
    train_auc["step"],
    train_auc["value"],
    label="Training AUC",
    color="#1f77b4",
    linewidth=2,
)
plt.plot(
    val_auc["step"],
    val_auc["value"],
    label="Validation AUC",
    color="#ff7f0e",
    linewidth=2,
)
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Training and Validation AUC over Epochs")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
epochs = np.array(sorted(set(train_auc["step"])))
plt.xticks(np.arange(epochs[0], epochs[-1] + 1, 5))
plt.savefig("auc_over_epochs_mask_032.png")

# Plot Loss over epochs
plt.figure(figsize=(8, 5))
plt.plot(
    train_loss["step"],
    train_loss["value"],
    label="Training Loss",
    color="#2ca02c",
    linewidth=2,
)
plt.plot(
    val_loss["step"],
    val_loss["value"],
    label="Validation Loss",
    color="#d62728",
    linewidth=2,
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
epochs = np.array(sorted(set(train_loss["step"])))
plt.xticks(np.arange(epochs[0], epochs[-1] + 1, 5))
plt.savefig("loss_over_epochs_mask_032.png")
