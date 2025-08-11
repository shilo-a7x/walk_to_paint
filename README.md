# Edge Classification via Random Walks + Transformers

This project performs masked edge classification on graph datasets. It converts graph edges into random walk sequences, tokenizes them, applies masking, and trains a Transformer-based model to predict edge labels.

---

## 📂 Project Structure

```
.
├── config.yaml                # Main config
├── run.py                    # Main entrypoint
├── requirements.txt
├── data/                     # Stores preprocessed data, walks, checkpoints, etc.
├── logs/                     # TensorBoard logs
├── src/
│   ├── data/
│   │   ├── datasets.py       # Dataset-specific graph loaders
│   │   ├── prepare_data.py   # Preprocessing pipeline
│   │   ├── tokenizer.py      # Tokenizer class for walks
│   │   └── walk_sampler.py   # Random walk sampling
│   ├── model/
│   │   ├── model.py          # Transformer model
│   │   └── lit_model.py      # PyTorch Lightning module
│   └── training/
│       └── train.py          # Training loop (Lightning)
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Configure settings

Edit `config.yaml` to specify:

-   Dataset name and file paths
-   Walk sampling parameters
-   Masking ratio
-   Training hyperparameters (batch size, learning rate, etc.)

Example:

```yaml
dataset:
    name: "chess"
    data_dir: "data/chess"
    edge_list_file: "data/chess/chess.out"
    walks_file: "data/chess/walks.json"
    tokenizer_file: "data/chess/tokenizer.json"
    encoded_file: "data/chess/encoded.pt"
    splits_file: "data/chess/splits.json"
    meta_file: "data/chess/meta.json"
    num_walks: 1500
    max_walk_length: 16
    mask_prob: 0.15
    ignore_index: -100
training:
    batch_size: 64
    epochs: 20
    lr: 1e-4
    use_cuda: true
    checkpoint_dir: "data/chess/checkpoints"
    resume_from_checkpoint: null
    eval_only: false
    log_dir: "logs"
    exp_name: "default"
preprocess:
    save: true
    use_cache: true
```

---

### 3. Prepare data and train

```bash
python run.py --config config.yaml
```

---

### 4. Resume from checkpoint

To continue training or evaluate a specific checkpoint:

```yaml
training:
    resume_from_checkpoint: "data/chess/checkpoints/chess-default-epoch=10-val_loss=0.15.ckpt"
```

---

### 5. Use CLI overrides (OmegaConf)

Override config from the command line:

```bash
python run.py --config config.yaml training.epochs=10 training.batch_size=32
```

---

### 6. Specify CUDA device

In terminal:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --config config.yaml
```

---

### 7. Adding a New Dataset

1. Add a loader in `src/data/datasets.py`:

```python
def load_new_dataset(cfg):
    # Return list of (u, v, label) tuples
    ...
DATASET_LOADERS = {
    "chess": load_chess,
    "bitcoin": load_bitcoin,
    "new": load_new_dataset,
}
```

2. Update paths in `config.yaml` for the new dataset.

3. Run as usual:

```bash
python run.py --config config.yaml dataset.name=new
```

---

## 📈 Outputs

-   **Logs** → `logs/<exp_name>` (for TensorBoard)
-   **Checkpoints** → `data/<dataset>/checkpoints/`
-   **Preprocessed** walks, tokenizer, splits → `data/<dataset>/`

---

## 🔍 Evaluation Metrics

-   Accuracy
-   F1 (macro)
-   Confusion Matrix

---

## 📎 Dependencies

-   PyTorch
-   PyTorch Lightning
-   OmegaConf
-   scikit-learn

---

## 🧠 Notes

-   Model input: padded sequences of tokenized walks
-   Output: per-token predictions over edge label space
-   Only edge tokens are masked and supervised
