# Edge Classification via Random Walks + Transformers

This project performs masked edge classification on graph datasets. It converts graph edges into random walk sequences, tokenizes them, applies masking, and trains a Transformer-based model to predict edge labels.

---

## 📂 Project Structure

```
.
├── config.yaml                # Main configuration file
├── config_toy.yaml            # Toy dataset configuration
├── run.py                     # Single run entrypoint
├── optuna_run.py              # Hyperparameter optimization with Optuna
├── extract_optuna_results.py  # Extract best trials from Optuna studies
├── extract_trials.py          # Analyze Optuna trial results
├── plot_metrics.py            # Visualize training metrics
├── launch_optuna.sh           # Script to launch Optuna studies
├── requirements.txt           # Python dependencies
├── configs/                   # Per-dataset configuration files
│   ├── toy.yaml
│   ├── bitcoin-alpha-binary.yaml
│   ├── wiki-rfa.yaml
│   ├── epinions.yaml
│   └── slashdot090221.yaml
├── outputs/                   # Organized outputs per dataset/experiment
├── src/
│   ├── data/
│   │   ├── datasets.py        # Dataset-specific graph loaders
│   │   ├── prepare_data.py    # Preprocessing pipeline
│   │   ├── tokenizer.py       # Tokenizer class for walks
│   │   └── walk_sampler.py    # Random walk sampling
│   ├── model/
│   │   ├── model.py           # Transformer model
│   │   └── lit_model.py       # PyTorch Lightning module
│   └── training/
│       └── train.py           # Training loop (Lightning)
├── scripts/                   # Utility scripts
└── tests/                     # Test suite
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Configure settings

The project uses a hierarchical configuration system with **config merging**:

1. **Base config** (`config.yaml`): Shared defaults
2. **Per-dataset configs** (`configs/<dataset>.yaml`): Dataset-specific settings
3. **CLI overrides**: Command-line arguments override both

#### Configuration Structure

```yaml
dataset:
    name: "toy"                    # Dataset identifier (merges configs/<name>.yaml)
    num_walks: 1500                # Number of walks per edge
    max_walk_length: 16            # Maximum walk length
    mask_prob: 0.15                # Masking probability for edges
    
training:
    batch_size: 64
    epochs: 20
    lr: 1e-4
    use_cuda: true
    
preprocess:
    save: true                     # Save preprocessed data
    use_cache: true                # Use cached data if available
    
paths:
    output_dir: "outputs"          # Root for all outputs
    append_timestamp: true         # Add timestamp to experiment folders
    
seed: 42                           # Global random seed for reproducibility
```

#### Supported Datasets

The project includes loaders for the following datasets:

- **toy**: Small synthetic dataset for testing
- **bitcoin-alpha-binary**: Bitcoin Alpha trust network (binary labels)
- **wiki-rfa**: Wikipedia Requests for Adminship
- **epinions**: Epinions social network
- **slashdot090221**: Slashdot social network

Each dataset has a corresponding config file in `configs/` with optimized parameters.

---

### 3. Running Experiments

#### Single Training Run

```bash
# Basic run with default config
python run.py --config config.yaml

# Run with specific dataset
python run.py --config config.yaml dataset.name=toy

# Override parameters from CLI
python run.py --config config.yaml dataset.name=bitcoin-alpha-binary training.epochs=30 training.lr=5e-4
```

#### Hyperparameter Optimization with Optuna

Run hyperparameter search using Optuna:

```bash
# Basic Optuna run
python optuna_run.py --config config.yaml --n-trials 50

# With specific dataset and storage
python optuna_run.py --config config.yaml --n-trials 100 dataset.name=wiki-rfa

# Extract best trials from study
python extract_optuna_results.py --study-dir outputs/bitcoin-alpha-binary/optuna_search/optuna/

# Analyze trials
python extract_trials.py
```

**Optuna Features:**
- Pruning callback for early stopping of unpromising trials
- Journal-based storage for persistence
- Automatic checkpoint management
- Best trial extraction and analysis

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

### 7. Output Organization

The project uses a structured output directory to organize results:

```
outputs/
└── <dataset_name>/
    └── <exp_name>[_<timestamp>]/
        ├── checkpoints/        # Model checkpoints
        ├── logs/               # TensorBoard logs
        ├── optuna/             # Optuna study files (journal, DB)
        └── plots/              # Generated plots and visualizations
```

**Benefits:**
- No accidental overwrites between experiments
- Easy comparison of results across datasets
- Organized storage for hyperparameter search results
- Optional timestamp suffix for multiple runs

**Control timestamp behavior:**
```yaml
paths:
    append_timestamp: false  # For deterministic folder names
```

### 8. Visualization and Analysis

Generate plots and analyze results:

```bash
# Plot training metrics from checkpoints
python plot_metrics.py

# Extract and visualize Optuna results
python extract_optuna_results.py --study-dir outputs/<dataset>/<exp>/optuna/
```

### 9. Adding a New Dataset

1. Add a loader in `src/data/datasets.py`:

```python
def load_new_dataset(cfg):
    # Load your graph data
    # Return list of (source, target, label) tuples
    edges = []
    # ... your loading logic ...
    return edges

DATASET_LOADERS = {
    "toy": load_toy,
    "bitcoin-alpha-binary": load_bitcoin_alpha,
    "wiki-rfa": load_wiki_rfa,
    "epinions": load_epinions,
    "slashdot090221": load_slashdot,
    "new_dataset": load_new_dataset,  # Add your loader
}
```

2. Create a config file `configs/new_dataset.yaml`:

```yaml
dataset:
    num_walks: 1000
    max_walk_length: 16
    # ... dataset-specific parameters ...
```

3. Run:

```bash
python run.py --config config.yaml dataset.name=new_dataset
```

---

## 📈 Reproducibility

The project ensures reproducible results through:

- **Unified random seed**: Single `seed` parameter in config controls all randomness sources
  - PyTorch Lightning seed
  - NumPy random state
  - Random walk generation
  - Data splitting
- **Deterministic algorithms**: PyTorch deterministic mode when possible
- **Version tracking**: Configuration files saved with outputs
- **Cached preprocessing**: Preprocessed data reused across runs when `preprocess.use_cache: true`

---

## 🔍 Evaluation Metrics

The model is evaluated using:

-   **Accuracy**: Overall classification accuracy
-   **F1 Score (macro)**: Macro-averaged F1 across classes
-   **AUC-ROC**: Area under the ROC curve (binary/multiclass)
-   **Confusion Matrix**: Per-class performance visualization
-   **Loss**: Cross-entropy loss on masked edge predictions

Metrics are logged to TensorBoard and tracked throughout training.

---

## 📎 Dependencies

Core dependencies:

-   **PyTorch**: Deep learning framework
-   **PyTorch Lightning**: Training loop management
-   **OmegaConf**: Hierarchical configuration management
-   **Optuna**: Hyperparameter optimization
-   **NetworkX**: Graph operations and random walk generation
-   **scikit-learn**: Metrics and data splitting
-   **TensorBoard**: Training visualization

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🧠 Technical Details

### Model Architecture
-   **Input**: Padded sequences of tokenized random walks
-   **Masking**: Only edge tokens are masked (not node tokens)
-   **Output**: Per-token predictions over edge label space
-   **Loss**: Computed only on masked edge positions

### Data Pipeline
1. **Graph Loading**: Load edge list with labels from dataset loaders
2. **Walk Sampling**: Generate random walks from graph structure
3. **Tokenization**: Convert walks to token sequences (nodes + edges)
4. **Masking**: Apply random masking to edge tokens
5. **Batching**: Pad sequences and create batches for training

### Performance Optimization
-   **Caching**: Preprocessed data cached to disk
-   **Parallel Loading**: Multi-worker data loading
-   **Mixed Precision**: Automatic mixed precision training support
-   **Early Stopping**: Based on validation metrics

---

## 📊 Example Workflow

```bash
# 1. Quick test on toy dataset
python run.py --config config.yaml dataset.name=toy training.epochs=5

# 2. Full training run
python run.py --config config.yaml dataset.name=bitcoin-alpha-binary

# 3. Hyperparameter search
python optuna_run.py --config config.yaml --n-trials 100 dataset.name=wiki-rfa

# 4. Extract best parameters
python extract_optuna_results.py --study-dir outputs/wiki-rfa/optuna_search/optuna/

# 5. Train with best parameters (copy from best_params_optuna.yaml)
python run.py --config config.yaml dataset.name=wiki-rfa training.lr=3e-4 model.d_model=128

# 6. Visualize results
python plot_metrics.py
tensorboard --logdir outputs/
```

---

## 🤝 Contributing

When adding new features:
1. Maintain the config merging system
2. Add dataset-specific configs to `configs/`
3. Update this README with usage examples
4. Ensure reproducibility with proper seeding

---

## 📝 License

[Add your license information here]
