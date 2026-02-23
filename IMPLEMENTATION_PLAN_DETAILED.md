# Complete Implementation Plan - Walk Metadata Tracking

## Current Status

**Current data pack structure:**

```python
train_pack = (train_x, train_y, train_attn)  # 3 tensors
val_pack = (val_x, val_y, val_attn)
test_pack = (test_x, test_y, test_attn)
```

**After modification:**

```python
train_pack = (train_x, train_y, train_attn, train_metadata)  # 4 tensors
val_pack = (val_x, val_y, val_attn, val_metadata)
test_pack = (test_x, test_y, test_attn, test_metadata)

# metadata is a dict with tensors:
metadata = {
    'edge_ids': torch.Tensor,     # Which edge (for aggregator)
    'walk_ids': torch.Tensor,     # Which walk
    'positions': torch.Tensor,    # Position in walk
    'walk_lengths': torch.Tensor, # Length of walk
}
```

---

## Implementation Steps

### Step 1: Track Walk Metadata During Encoding

**File:** `src/data/prepare_data.py`
**Function:** `encode_walks()`

**Current:**

```python
def encode_walks(walks, tokenizer, train_set, mask_set, val_set, test_set):
    input_ids_list = []
    edge_split_masks_list = []
    
    for walk in walks:
        # encode tokens
        input_ids_list.append(ids)
        edge_split_masks_list.append(splits)
    
    return input_ids_list, edge_split_masks_list
```

**New:**

```python
def encode_walks(walks, tokenizer, train_set, mask_set, val_set, test_set):
    input_ids_list = []
    edge_split_masks_list = []
    
    # NEW: Track walk metadata
    edge_ids_list = []      # Which edge at each position
    walk_ids_list = []      # Which walk (just index)
    positions_list = []     # Position in walk
    walk_lengths_list = []  # Length of walk
    
    for walk_idx, walk in enumerate(walks):
        # encode tokens
        input_ids_list.append(ids)
        edge_split_masks_list.append(splits)
        
        # NEW: Extract edge_ids from walk
        edge_ids = []
        for token in walk:
            if tokenizer.is_edge(token):
                src, dst, label = tokenizer.parse_edge(token)
                edge_id = get_edge_id(src, dst)  # Need edge lookup
                edge_ids.append(edge_id)
            else:
                edge_ids.append(-1)  # Not an edge position
        
        edge_ids_list.append(torch.tensor(edge_ids))
        walk_ids_list.append(torch.full((len(walk),), walk_idx))
        positions_list.append(torch.arange(len(walk)))
        walk_lengths_list.append(torch.full((len(walk),), len(walk)))
    
    return (input_ids_list, edge_split_masks_list, 
            edge_ids_list, walk_ids_list, positions_list, walk_lengths_list)
```

### Step 2: Build Stage-Specific Metadata Views

**File:** `src/data/prepare_data.py`
**Function:** `_stage_views_from_base()`

**New:**

```python
def _stage_views_from_base(
    input_ids, edge_split_mask, tokenizer,
    edge_ids, walk_ids, positions, walk_lengths  # NEW PARAMS
):
    # ... existing logic for x, y, attn ...
    
    # Build metadata dict for each stage
    train_metadata = {
        'edge_ids': edge_ids,
        'walk_ids': walk_ids,
        'positions': positions,
        'walk_lengths': walk_lengths,
    }
    
    val_metadata = {  # Same for val
        'edge_ids': edge_ids,
        'walk_ids': walk_ids,
        'positions': positions,
        'walk_lengths': walk_lengths,
    }
    
    test_metadata = {  # Same for test
        'edge_ids': edge_ids,
        'walk_ids': walk_ids,
        'positions': positions,
        'walk_lengths': walk_lengths,
    }
    
    return (
        (train_x, train_y, train_attn, train_metadata),
        (val_x, val_y, val_attn, val_metadata),
        (test_x, test_y, test_attn, test_metadata),
    )
```

### Step 3: Custom Dataset Class

**Why?** `TensorDataset` only handles tensors, not dicts.

**File:** `src/data/walk_dataset.py` (NEW)

```python
class WalkDataset(Dataset):
    """Dataset with walk metadata support."""
    
    def __init__(self, input_ids, labels, attention_mask, metadata):
        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask
        self.metadata = metadata
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.labels[idx],
            self.attention_mask[idx],
            {k: v[idx] for k, v in self.metadata.items()}
        )
```

### Step 4: Update make_dataloaders

**File:** `src/data/prepare_data.py`

```python
def make_dataloaders(cfg, train_pack, val_pack, test_pack):
    # Unpack with metadata
    train_x, train_y, train_attn, train_meta = train_pack
    val_x, val_y, val_attn, val_meta = val_pack
    test_x, test_y, test_attn, test_meta = test_pack
    
    # Create WalkDataset instead of TensorDataset
    train_ds = WalkDataset(train_x, train_y, train_attn, train_meta)
    val_ds = WalkDataset(val_x, val_y, val_attn, val_meta)
    test_ds = WalkDataset(test_x, test_y, test_attn, test_meta)
    
    # ... rest remains the same ...
```

---

## What This Gives Us

After training, we have saved per epoch:

```python
# checkpoints/wiki-rfa_predictions/epoch_000/train_predictions.pkl
{
    'epoch': 0,
    'split': 'train',
    'edge_ids': [42, 99, 42, ...],      # Can group by edge_id
    'walk_ids': [0, 0, 1, ...],         # Which walk
    'positions': [0, 3, 1, ...],        # Position in walk
    'walk_lengths': [10, 10, 12, ...],  # Walk length
    'dist_from_start': [0, 3, 1, ...],  # = positions
    'dist_from_end': [9, 6, 10, ...],   # = walk_length - position - 1
    'predictions': [1, 0, 1, ...],      # Predicted labels
    'probabilities': [[0.2, 0.8], ...], # Softmax probs
    'targets': [1, 0, 1, ...],          # True labels
    'correct': [True, True, True, ...], # Correctness
    'auc': 0.85,                        # Split AUC
}
```

---

## Post-Training Analysis (NO MODEL LOADING!)

### For Heatmaps

```python
predictions = load_pickle(f"epoch_{i}/val_predictions.pkl")

# Already have triplets!
triplets = {
    'dist_from_start': predictions['dist_from_start'],
    'dist_from_end': predictions['dist_from_end'],
    'correct': predictions['correct'],
}

# Plot heatmap
plot_heatmap(triplets)  # No model inference!
```

### For Aggregator Training

```python
predictions = load_pickle(f"epoch_{i}/val_predictions.pkl")

# Group by edge_id
from collections import defaultdict
edge_preds = defaultdict(list)

for i, edge_id in enumerate(predictions['edge_ids']):
    edge_preds[edge_id].append({
        'prob': predictions['probabilities'][i],
        'pred': predictions['predictions'][i],
        'target': predictions['targets'][i],
        'position': predictions['positions'][i],
        'walk_length': predictions['walk_lengths'][i],
    })

# Train aggregator
for edge_id, occurrences in edge_preds.items():
    # Multiple predictions per edge
    # Aggregate them (mean, voting, etc.)
    pass
```

### For Walk-Length Analysis

```python
# Analyze accuracy by walk length
import numpy as np

lengths = predictions['walk_lengths']
correct = predictions['correct']

for length in np.unique(lengths):
    mask = lengths == length
    acc = np.mean(correct[mask])
    print(f"Walk length {length}: accuracy = {acc:.3f}")
```

---

## Key Challenge: Edge ID Lookup

**Problem:** During encoding, we have tokens like `"(42,99,+)"` but need to map back to original edge_id.

**Solution:** Pass edge list through pipeline:

```python
def encode_walks(walks, tokenizer, edges, train_set, mask_set, val_set, test_set):
    # Build edge lookup
    edge_to_id = {}
    for edge_id, (src, dst, label, timestamp) in enumerate(edges):
        edge_to_id[(src, dst)] = edge_id
    
    # During encoding
    for token in walk:
        if tokenizer.is_edge(token):
            src, dst, label = tokenizer.parse_edge(token)
            edge_id = edge_to_id.get((src, dst), -1)
```

This requires propagating `edges` through the preprocessing pipeline.

---

## Files to Modify

1. **src/data/prepare_data.py**
   - `encode_walks()`: Track walk metadata
   - `pad_and_build_stage_tensors()`: Pad metadata tensors
   - `_stage_views_from_base()`: Return metadata with each pack
   - `make_dataloaders()`: Use WalkDataset instead of TensorDataset

2. **src/data/walk_dataset.py** (NEW)
   - `WalkDataset` class

3. **src/training/callbacks.py** (ALREADY DONE)
   - `PerEpochPredictionSaver` handles metadata

4. **src/training/train.py** (ALREADY DONE)
   - Registers callbacks

---

## Backward Compatibility

**Problem:** Cached encoded.pt files don't have metadata.

**Solution:**

```python
if cfg.preprocess.use_cache and os.path.exists(enc_path):
    loaded = torch.load(enc_path)
    
    # Check if old format (3-tuple) or new format (4-tuple)
    if len(loaded[0]) == 3:
        # Old format - create dummy metadata
        train_x, train_y, train_attn = loaded[0]
        train_meta = create_dummy_metadata(train_x.shape)
        train_pack = (train_x, train_y, train_attn, train_meta)
        # ... same for val, test ...
    else:
        # New format - has metadata
        train_pack, val_pack, test_pack = loaded
```

---

## Testing Plan

1. **Test data pipeline:**

   ```bash
   python -c "
   from src.utils.config import load_config
   from src.data.prepare_data import prepare_data
   cfg = load_config('config.yaml', ['dataset.name=bitcoin-alpha-binary'])
   data_module = prepare_data(cfg)
   batch = next(iter(data_module['train']))
   assert len(batch) == 4  # (x, y, attn, metadata)
   print('✓ Pipeline works!')
   "
   ```

2. **Test training with callbacks:**

   ```bash
   python run.py --config configs/wiki-rfa.yaml training.epochs=2
   # Check: checkpoints/wiki-rfa_predictions/epoch_000/*.pkl exist
   ```

3. **Test predictions loading:**

   ```python
   import pickle
   with open('checkpoints/wiki-rfa_predictions/epoch_000/val_predictions.pkl', 'rb') as f:
       preds = pickle.load(f)
   
   assert 'edge_ids' in preds
   assert 'dist_from_start' in preds
   print('✓ Predictions have all metadata!')
   ```
