from tbparse import SummaryReader
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

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


# --- Plot final ROC curve from TensorBoard images (if available) ---
def save_last_roc_image(df, tag, out_path):
    # Find all image events for the given tag
    img_df = df[(df["tag"] == tag) & (df["type"] == "image")]
    if len(img_df) == 0:
        print(f"No ROC curve images found for tag: {tag}")
        return
    # Get the last ROC curve image (highest step)
    last_img = img_df.sort_values("step").iloc[-1]
    img_data = last_img["value"]  # This is a numpy array (H, W, 3) or (H, W, 4)
    # Convert to uint8 if needed
    if img_data.dtype != np.uint8:
        img_data = (img_data * 255).astype(np.uint8)
    # Remove alpha channel if present
    if img_data.shape[-1] == 4:
        img_data = img_data[..., :3]
    img = Image.fromarray(img_data)
    img.save(out_path)
    print(f"Saved ROC curve image: {out_path}")


# Save final ROC curves for train and validation
save_last_roc_image(df, "train_roc_curve", "final_train_roc_curve.png")
save_last_roc_image(df, "val_roc_curve", "final_val_roc_curve.png")
